"""Reconcile retained direct clock outcomes without reissuing their generation.

Two ordinary workflows deliberately separate new evidence production from clock
adoption. Only replay.workflow writes execution state. Historical boundaries that
were never captured remain unknown even after current input bytes are verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from replay.audio_clock import AudioFrameClock
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.delivery_media import MAX_CLOCK_BYTES, parse_clock
from replay.recordings import record
from replay.sha256_check import digest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "retained-clock-adoption-v1"
MAX_METADATA_BYTES = 1024 * 1024
MAX_PHASE_BYTES = 16 * 1024 * 1024
MAX_INPUTS = 220
IDENTITY_FIELDS = frozenset({"dev", "inode", "bytes", "mtimeNs", "ctimeNs", "nlink"})
IDENTITY_FIELDS_ORDER = ("dev", "inode", "bytes", "mtimeNs", "ctimeNs", "nlink")
CONFIG_FIELDS = frozenset(
    {
        "version",
        "protocol",
        "source",
        "tools",
        "clocks",
        "phases",
        "dispatch",
        "outcome",
        "codeAfterDispatch",
        "codeTerminal",
        "sourceIdentityEvidence",
        "sourceDigestEvidence",
        "timeoutSeconds",
    }
)
EVIDENCE_FIELDS = (
    "phases",
    "dispatch",
    "outcome",
    "codeAfterDispatch",
    "codeTerminal",
    "sourceIdentityEvidence",
    "sourceDigestEvidence",
)
Kind = Literal["video", "audio"]


@dataclass(frozen=True, slots=True)
class Pin:
    path: Path
    sha256: str

    def data(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class Tool:
    invocation: Path
    pin: Pin


@dataclass(frozen=True, slots=True)
class Clock:
    kind: Kind
    output: Pin
    status: Pin
    log: Pin
    job: str
    phase: str


@dataclass(frozen=True, slots=True)
class Job:
    config: Pin
    source: Pin
    source_identity: dict[str, int]
    tools: dict[str, Tool]
    clocks: tuple[Clock, ...]
    evidence: dict[str, Pin]
    timeout: int


@dataclass(frozen=True, slots=True)
class Prepared:
    evidence_workflow: Path
    adoption_workflow: Path
    sequence: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _path(value: object, *, local: bool = True) -> Path:
    _require(isinstance(value, str) and bool(value) and "\0" not in value, "Invalid path")
    path = Path(str(value))
    _require(
        path.is_absolute()
        and str(path) == value
        and ".." not in path.parts
        and path == path.resolve(),
        "Expected a canonical unlinked path",
    )
    if local:
        _require(path.is_relative_to(ROOT / "edits" / "work"), "Evidence escapes repository work")
    return path


def _identity(path: Path) -> dict[str, int]:
    _path(str(path), local=False)
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Expected single-link regular file")
    return dict(
        zip(
            IDENTITY_FIELDS_ORDER,
            (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
                info.st_nlink,
            ),
            strict=True,
        )
    )


def _bytes(path: Path, budget: int) -> bytes:
    before = _identity(path)
    _require(before["bytes"] <= budget, "Evidence exceeds byte budget")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        opened = os.fstat(stream.fileno())
        _require(
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            == tuple(before[key] for key in IDENTITY_FIELDS_ORDER[:-1]),
            "Evidence changed before reading",
        )
        data = stream.read(budget + 1)
        _require(
            len(data) == before["bytes"] and _identity(path) == before,
            "Evidence changed while reading",
        )
    return data


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in items:
        _require(name not in result, "Duplicate JSON key")
        result[name] = value
    return result


def _json(path: Path, *, budget: int = MAX_METADATA_BYTES) -> dict[str, object]:
    return record(json.loads(_bytes(path, budget), object_pairs_hook=_pairs))


def _pin(value: object) -> Pin:
    raw = record(value)
    _require(set(raw) == {"path", "sha256"}, "Invalid pin fields")
    sha = raw["sha256"]
    _require(
        isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha) is not None, "Invalid SHA256"
    )
    return Pin(_path(raw["path"]), str(sha))


def _tool(value: object) -> Tool:
    raw = record(value)
    _require(set(raw) == {"path", "sha256"}, "Invalid tool fields")
    text = raw["path"]
    _require(isinstance(text, str) and bool(text), "Missing tool invocation")
    invocation = Path(str(text))
    _require(
        invocation.is_absolute() and str(invocation) == text and ".." not in invocation.parts,
        "Invalid tool invocation",
    )
    target = _path(str(invocation.resolve(strict=True)), local=False)
    sha = raw["sha256"]
    _require(
        isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha) is not None,
        "Invalid tool SHA256",
    )
    _require(os.access(invocation, os.X_OK), "Tool must be executable")
    _identity(target)
    return Tool(invocation, Pin(target, str(sha)))


def _clock(value: object) -> Clock:
    raw = record(value)
    _require(
        set(raw) == {"kind", "output", "status", "log", "job", "phase"}, "Invalid clock fields"
    )
    kind = raw["kind"]
    _require(kind in ("video", "audio"), "Unknown clock kind")
    job, phase = raw["job"], raw["phase"]
    _require(
        isinstance(job, str) and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", job) is not None,
        "Invalid original job",
    )
    _require(isinstance(phase, str) and 0 < len(phase) <= 256, "Invalid original phase")
    return Clock(
        "video" if kind == "video" else "audio",
        _pin(raw["output"]),
        _pin(raw["status"]),
        _pin(raw["log"]),
        str(job),
        str(phase),
    )


def _job(config: Path) -> Job:
    _path(str(config))
    raw = _json(config)
    _require(
        set(raw) == CONFIG_FIELDS
        and type(raw["version"]) is int
        and raw["version"] == 1
        and raw["protocol"] == PROTOCOL,
        "Invalid adoption configuration",
    )
    source = record(raw["source"])
    _require(set(source) == {"path", "sha256", "identity"}, "Invalid source fields")
    identity = record(source["identity"])
    _require(
        set(identity) == IDENTITY_FIELDS
        and all(type(value) is int and value > 0 for value in identity.values())
        and identity["nlink"] == 1,
        "Invalid retained source identity",
    )
    tools = record(raw["tools"])
    _require(set(tools) == {"python", "ffprobe", "env"}, "Missing or unknown tool role")
    raw_clocks = raw["clocks"]
    _require(isinstance(raw_clocks, list) and len(raw_clocks) == 2, "Expected both original clocks")
    if not isinstance(raw_clocks, list):
        raise ReplayError("Invalid clocks")
    clocks = tuple(_clock(value) for value in raw_clocks)
    _require(
        tuple(clock.kind for clock in clocks) == ("video", "audio")
        and len({clock.phase for clock in clocks}) == 2
        and len({clock.job for clock in clocks}) == 2,
        "Duplicate or unordered clock roles",
    )
    timeout = raw["timeoutSeconds"]
    _require(type(timeout) is int and 1 <= timeout <= 86400, "Invalid adoption deadline")
    return Job(
        Pin(config, hashlib.sha256(_bytes(config, MAX_METADATA_BYTES)).hexdigest()),
        _pin({"path": source["path"], "sha256": source["sha256"]}),
        {key: int(str(value)) for key, value in identity.items()},
        {key: _tool(value) for key, value in tools.items()},
        clocks,
        {key: _pin(raw[key]) for key in EVIDENCE_FIELDS},
        int(str(timeout)),
    )


def _code(modules: tuple[str, ...]) -> tuple[Pin, ...]:
    return tuple(
        Pin(item.path, item.sha256) for item in capture(modules, package=ROOT / "src" / "replay")
    )


def _expected(job: Job) -> tuple[Pin, ...]:
    entries = (
        job.config,
        job.source,
        *job.evidence.values(),
        *(tool.pin for tool in job.tools.values()),
        *(pin for clock in job.clocks for pin in (clock.output, clock.status, clock.log)),
        *_code(
            ("replay_clock_adoption", "delivery_media", "phase_timer", "code_dependency_inventory")
        ),
    )
    found: dict[Path, str] = {}
    for pin in entries:
        _require(pin.path not in found or found[pin.path] == pin.sha256, "Conflicting input pins")
        found[pin.path] = pin.sha256
    _require(len(found) <= MAX_INPUTS, "Adoption input budget exceeded")
    outputs = {clock.output.path for clock in job.clocks}
    _require(
        len(outputs) == 2 and job.source.path not in outputs, "Overlapping source and clock roles"
    )
    # Keep empty process logs pinned and checked by this reader; stages deliberately
    # rejects empty direct inputs. Their transitive byte pins remain in the plan.
    return tuple(Pin(path, sha) for path, sha in sorted(found.items()))


def _write(path: Path, value: object) -> None:
    _path(str(path))
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600),
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _argv(job: Job, action: str, plan: Path, receipt: Path) -> list[str]:
    work = plan.parent
    return [
        str(job.tools["env"].invocation),
        f"TMPDIR={work / 'scratch'}",
        f"TMP={work / 'scratch'}",
        f"TEMP={work / 'scratch'}",
        f"XDG_CACHE_HOME={work / 'cache'}",
        f"PYTHONPATH={ROOT / 'src'}",
        "PYTHONDONTWRITEBYTECODE=1",
        str(job.tools["python"].invocation),
        "-B",
        "-m",
        "replay.replay_clock_adoption",
        action,
        str(plan),
        str(receipt),
    ]


def prepare(config: Path, work: Path) -> Prepared:
    """Write two non-overwriting graphs; no source hashing, decoder or adoption runs."""
    _path(str(work))
    _require(work.parent.is_dir() and not work.exists(), "Use a new reconciliation work root")
    job = _job(config)
    pins = _expected(job)
    plan, receipt = work / "plan.json", work / "adoption-evidence.json"
    clock_paths = {clock.output.path for clock in job.clocks}
    direct = [str(pin.path) for pin in pins if _identity(pin.path)["bytes"] > 0]
    evidence = {
        "version": 1,
        "stages": [
            {
                "name": "seal-retained-clocks",
                "command": _argv(job, "seal", plan, receipt),
                "verify_command": _argv(job, "verify", plan, receipt),
                "inputs": [*direct, str(plan)],
                "outputs": [str(receipt)],
                "timeout_seconds": job.timeout,
            }
        ],
    }
    adoption = {
        "version": 1,
        "stages": [
            {
                "name": "adopt-retained-clocks",
                "command": _argv(job, "verify", plan, receipt),
                "verify_command": _argv(job, "verify", plan, receipt),
                "inputs": [
                    *(path for path in direct if Path(path) not in clock_paths),
                    str(plan),
                    str(receipt),
                ],
                "outputs": [str(clock.output.path) for clock in job.clocks],
                "timeout_seconds": job.timeout,
                "adopt_existing": True,
            }
        ],
    }
    work.mkdir(mode=0o700)
    (work / "scratch").mkdir(mode=0o700)
    (work / "cache").mkdir(mode=0o700)
    _write(
        plan,
        {
            "version": 1,
            "protocol": PROTOCOL,
            "config": job.config.data(),
            "inputs": [pin.data() for pin in pins],
        },
    )
    paths = Prepared(
        work / "evidence-workflow.json", work / "adoption-workflow.json", work / "sequence.json"
    )
    _write(paths.evidence_workflow, evidence)
    _write(paths.adoption_workflow, adoption)
    _write(
        paths.sequence,
        {
            "version": 1,
            "protocol": PROTOCOL,
            "order": [
                {
                    "workflow": str(paths.evidence_workflow),
                    "state": str(work / "evidence-state"),
                    "requiredExit": 0,
                },
                {
                    "workflow": str(paths.adoption_workflow),
                    "state": str(work / "adoption-state"),
                    "requiredExit": 0,
                },
            ],
            "originalGenerationReissued": False,
            "deliveryComplete": False,
        },
    )
    return paths


def _load(plan: Path) -> tuple[Job, tuple[Pin, ...]]:
    _path(str(plan))
    raw = _json(plan)
    _require(
        set(raw) == {"version", "protocol", "config", "inputs"}
        and type(raw["version"]) is int
        and raw["version"] == 1
        and raw["protocol"] == PROTOCOL,
        "Invalid adoption plan",
    )
    config = _pin(raw["config"])
    _require(digest(config.path) == config.sha256, "Configuration pin changed")
    job = _job(config.path)
    items = raw["inputs"]
    _require(isinstance(items, list) and 1 <= len(items) <= MAX_INPUTS, "Invalid input closure")
    if not isinstance(items, list):
        raise ReplayError("Invalid input closure")
    # Tool and code paths are outside the evidence-only path parser by design.
    pins: list[Pin] = []
    for value in items:
        item = record(value)
        _require(set(item) == {"path", "sha256"}, "Invalid input fields")
        sha = item["sha256"]
        _require(
            isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha) is not None,
            "Invalid input SHA256",
        )
        pins.append(Pin(_path(item["path"], local=False), str(sha)))
    _require(tuple(pins) == _expected(job), "Input membership or code pins changed")
    return job, tuple(pins)


def _fresh(pins: tuple[Pin, ...]) -> dict[Path, dict[str, int]]:
    identities: dict[Path, dict[str, int]] = {}
    for pin in pins:
        before = _identity(pin.path)
        _require(
            digest(pin.path) == pin.sha256 and _identity(pin.path) == before, "Input bytes changed"
        )
        identities[pin.path] = before
    _require(
        all(_identity(path) == identity for path, identity in identities.items()),
        "Input identity changed",
    )
    return identities


def _phase(job: Job, clock: Clock) -> dict[str, object]:
    data = _bytes(job.evidence["phases"].path, MAX_PHASE_BYTES)
    _require(data.endswith(b"\n"), "Incomplete phase log")
    rows = [record(json.loads(line, object_pairs_hook=_pairs)) for line in data.splitlines()]
    matches = [row for row in rows if row.get("phase") == clock.phase]
    _require(len(matches) == 1, "Original phase must be unique")
    row = matches[0]
    _require(
        set(row) == {"phase", "start", "end", "seconds", "exit"}
        and type(row["exit"]) is int
        and row["exit"] == 0,
        "Original phase failed or has invalid fields",
    )
    times = [row[key] for key in ("start", "end", "seconds")]
    _require(
        all(type(value) in (int, float) and math.isfinite(float(str(value))) for value in times),
        "Invalid original phase time",
    )
    start, end, seconds = (float(str(value)) for value in times)
    # phase_timer records diagnostic epochs with time.time(), but elapsed seconds
    # with time.monotonic(). Wall-clock adjustments need not match elapsed time.
    # _domains separately binds the recorded elapsed value to the original result.
    _require(0 < start <= end and 0 <= seconds <= job.timeout, "Original phase timing differs")
    _require(_bytes(clock.status.path, 64) == b"0\n", "Original status is not successful")
    _require(
        _bytes(clock.log.path, MAX_METADATA_BYTES) == b"",
        "Original clock log contains unreviewed diagnostics",
    )
    return row


def _summary(clock: Clock) -> dict[str, object]:
    data = _json(clock.output.path, budget=MAX_CLOCK_BYTES)
    if clock.kind == "audio":
        audio = AudioFrameClock.from_probe(data)
        return {
            "frameCount": len(audio.pts),
            "sampleRate": audio.sample_rate,
            "timeBase": str(audio.time_base),
            "firstPTS": audio.pts[0],
            "lastPTS": audio.pts[-1],
            "decodedSamples": audio.sample_ends[-1],
        }
    video = parse_clock(data)
    frames = data.get("frames")
    _require(
        isinstance(frames, list)
        and all(
            type(record(row).get("duration")) is int and int(str(record(row)["duration"])) > 0
            for row in frames
        )
        and video.time_base <= 1,
        "Invalid decoded video durations or time base",
    )
    return {
        "frameCount": len(video.pts),
        "timeBase": str(video.time_base),
        "firstPTS": video.pts[0],
        "lastPTS": video.pts[-1],
        "lastDuration": video.last_duration,
    }


def _domains(job: Job) -> tuple[dict[str, object], dict[str, str]]:
    after = record(_json(job.evidence["codeAfterDispatch"].path).get("inputs"))
    terminal = record(_json(job.evidence["codeTerminal"].path).get("inputs"))
    current = {
        str(pin.path): pin.sha256
        for pin in _code(("delivery_media", "phase_timer", "code_dependency_inventory"))
    }
    _require(after == terminal == current, "After-dispatch producer code changed")
    dispatch = _json(job.evidence["dispatch"].path)
    outcome = _json(job.evidence["outcome"].path)
    _require(
        dispatch.get("source") == str(job.source.path)
        and dispatch.get("producer") == "replay.delivery_media"
        and dispatch.get("initialTransitiveCodeSnapshotCaptured") is False
        and dispatch.get("sourceDigestCapturedAtDispatch") is False,
        "Historical dispatch limitations changed",
    )
    _require(
        outcome.get("initialCodeSnapshotBeforeDispatch") is False
        and outcome.get("domainChecksPerformedByOriginalProducer") is True
        and outcome.get("afterDispatchToTerminalSourceClosureEqual") is True
        and outcome.get("sourceSHA256") == job.source.sha256
        and outcome.get("clockOffsetsModified") is False
        and outcome.get("synchronizationApproved") is False
        and outcome.get("deliveryComplete") is False,
        "Historical outcome or limitations changed",
    )
    identity = record(_json(job.evidence["sourceIdentityEvidence"].path).get("source"))
    _require(
        identity.get("path") == str(job.source.path)
        and {key: identity.get(key) for key in IDENTITY_FIELDS} == job.source_identity
        and _identity(job.source.path) == job.source_identity,
        "Retained source identity changed",
    )
    text = _bytes(job.evidence["sourceDigestEvidence"].path, MAX_METADATA_BYTES).decode("utf-8")
    matches = [
        line for line in text.splitlines() if line.partition("  ")[2] == str(job.source.path)
    ]
    _require(
        matches == [f"{job.source.sha256}  {job.source.path}"],
        "Retained source digest evidence differs",
    )
    summaries: dict[str, object] = {}
    jobs = record(dispatch.get("jobs"))
    for clock in job.clocks:
        phase = _phase(job, clock)
        summary = _summary(clock)
        expected = {
            "job": clock.job,
            "exit": 0,
            "seconds": phase["seconds"],
            "sha256": clock.output.sha256,
            **summary,
        }
        _require(
            jobs.get(clock.kind) == clock.job and record(outcome.get(clock.kind)) == expected,
            "Original clock outcome differs",
        )
        summaries[clock.kind] = {"phase": phase, "summary": summary}
    return summaries, current


def _verified(plan: Path) -> dict[str, object]:
    plan_pin = Pin(plan, digest(plan))
    job, pins = _load(plan)
    bindings = (*pins, plan_pin)
    identities = _fresh(bindings)
    summaries, producer = _domains(job)
    reader_code = {str(pin.path): pin.sha256 for pin in _code(("replay_clock_adoption",))}
    final = _fresh(bindings)
    _require(
        final == identities and _identity(job.source.path) == job.source_identity,
        "Inputs changed during adoption verification",
    )
    _require(
        all(tool.invocation.resolve(strict=True) == tool.pin.path for tool in job.tools.values()),
        "Tool alias changed",
    )
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "kind": "limited-current-input-retained-outcome",
        "inputs": {str(pin.path): pin.sha256 for pin in bindings},
        "outputs": {str(clock.output.path): clock.output.sha256 for clock in job.clocks},
        "sourceIdentity": job.source_identity,
        "clocks": summaries,
        "historicalProducerCodeAfterDispatch": producer,
        "currentReaderCode": reader_code,
        "currentInputBytesVerified": True,
        "currentClockDomainsVerified": True,
        "historicalInitialCodeSnapshotVerified": False,
        "historicalSourceDigestAtDispatchVerified": False,
        "historicalOriginalArgvRecorded": False,
        "historicalToolPinsAtDispatchVerified": False,
        "historicalGenerationBindingFullyVerified": False,
        "historicalSourceToClockBindingVerified": False,
        "fullSourceClockCoverageIndependentlyVerified": False,
        "originalGenerationReceiptReissued": False,
        "initialCodeSnapshotBeforeDispatch": False,
        "sourceDigestCapturedAtDispatch": False,
        "clockOffsetsModified": False,
        "headPlacementApproved": False,
        "synchronizationApproved": False,
        "qualityApproved": False,
        "candidateOnly": True,
        "deliveryComplete": False,
    }


def seal(plan: Path, receipt: Path) -> None:
    """Create new evidence after current verification; never execute the old producer."""
    _path(str(receipt))
    _require(
        receipt.parent == plan.parent and not receipt.exists(),
        "Use an absent evidence receipt in the plan root",
    )
    result = _verified(plan)
    _write(receipt, result)


def verify(plan: Path, receipt: Path) -> None:
    """Read-only fresh revalidation; an existing receipt never gets rewritten."""
    _path(str(receipt))
    _require(receipt.parent == plan.parent, "Evidence receipt is outside its plan root")
    before = _bytes(receipt, MAX_METADATA_BYTES)
    _require(
        record(json.loads(before, object_pairs_hook=_pairs)) == _verified(plan),
        "Adoption evidence changed",
    )
    _require(
        _bytes(receipt, MAX_METADATA_BYTES) == before,
        "Evidence receipt changed during verification",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "seal", "verify"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            print(prepare(args.input, args.output).sequence)
        else:
            {"seal": seal, "verify": verify}[args.action](args.input, args.output)
    except (ReplayError, OSError, ValueError, SyntaxError) as error:
        print(f"CLOCK_ADOPTION_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
