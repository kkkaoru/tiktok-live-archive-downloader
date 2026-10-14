"""Adopt a retained separation outcome after fresh, source-bound verification.

The original direct process is never replayed or given a fabricated generation
receipt. Two ordinary workflows first seal current evidence, then adopt only the
original vocals and result through replay.workflow's normal state machinery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay import replay_clock_adoption as retained
from replay.audio_clock import AudioFrameClock
from replay.bgm_separation import MODEL_CONFIG_SHA256, MODEL_NAME, MODEL_REVISION, MODEL_SHA256
from replay.core import ReplayError
from replay.delivery_audio import verify_material
from replay.delivery_media import MAX_CLOCK_BYTES
from replay.recordings import record
from replay.sha256_check import digest
from replay.stages import Stage, StageError, _fingerprint
from replay.workflow import parse_stages

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "retained-separation-adoption-v1"
ROLES = frozenset({"source", "modelInput", "vocals", "result", "audioClock"})
BOUNDARIES = frozenset(
    {
        "codeBefore",
        "codeTerminal",
        "modelBefore",
        "modelTerminal",
        "providerBefore",
        "providerTerminal",
    }
)
CONFIG_FIELDS = frozenset(
    {
        "version",
        "protocol",
        "roles",
        "tools",
        "model",
        "providerInputs",
        "boundaries",
        "terminal",
        "clockStage",
        "timeoutSeconds",
    }
)
PRODUCER_MODULES = ("delivery_audio", "bgm_separation", "phase_timer", "code_dependency_inventory")


@dataclass(frozen=True, slots=True)
class Job:
    config: retained.Pin
    roles: dict[str, retained.Pin]
    tools: dict[str, retained.Tool]
    model: tuple[retained.Pin, retained.Pin]
    provider: tuple[retained.Pin, ...]
    boundaries: dict[str, Boundary]
    terminal: dict[str, retained.Pin]
    phase: str
    clock_workflow: retained.Pin
    clock_receipt: retained.Pin
    clock_inputs: tuple[retained.Pin, ...]
    clock_stage: Stage
    timeout: int
    supporting: tuple[retained.Pin, ...]


@dataclass(frozen=True, slots=True)
class Boundary:
    evidence: retained.Pin
    inventory: bool


def _path(value: object, *, local: bool = True) -> Path:
    path = retained._path(value, local=False)
    if local:
        retained._require(
            path.is_relative_to(ROOT / "edits" / "work"), "Evidence escapes repository work"
        )
    return path


def _pin(value: object, *, local: bool = True) -> retained.Pin:
    data = record(value)
    retained._require(set(data) == {"path", "sha256"}, "Invalid pin fields")
    sha = data["sha256"]
    retained._require(
        isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha) is not None, "Invalid SHA256"
    )
    return retained.Pin(_path(data["path"], local=local), str(sha))


def _pins(value: object, *, maximum: int) -> tuple[retained.Pin, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ReplayError("Expected bounded input pins")
    result = tuple(_pin(item, local=False) for item in value)
    retained._require(len({pin.path for pin in result}) == len(result), "Duplicate input pins")
    return result


def _boundary(value: object) -> Boundary:
    data = record(value)
    retained._require(
        set(data) == {"evidence", "format"}
        and data["format"] in ("input-inventory", "retained-metadata"),
        "Invalid boundary evidence format",
    )
    return Boundary(_pin(data["evidence"]), data["format"] == "input-inventory")


def _job(config: Path) -> Job:
    _path(str(config))
    data = retained._json(config)
    retained._require(
        CONFIG_FIELDS <= set(data) <= CONFIG_FIELDS | {"supportingEvidence"}
        and type(data["version"]) is int
        and data["version"] == 1
        and data["protocol"] == PROTOCOL,
        "Invalid separation adoption configuration",
    )
    raw_roles = record(data["roles"])
    retained._require(set(raw_roles) == ROLES, "Missing or unknown separation role")
    roles = {key: _pin(value, local=key != "source") for key, value in raw_roles.items()}
    retained._require(
        len({pin.path for pin in roles.values()}) == len(ROLES), "Overlapping separation roles"
    )
    tools = record(data["tools"])
    retained._require(set(tools) == {"python", "env"}, "Missing or unknown verification tool")
    model = record(data["model"])
    retained._require(
        set(model) == {"name", "revision", "weights", "config"}
        and model["name"] == MODEL_NAME
        and model["revision"] == MODEL_REVISION,
        "Wrong separation model",
    )
    weights, model_config = _pin(model["weights"], local=False), _pin(model["config"], local=False)
    retained._require(
        weights.sha256 == MODEL_SHA256
        and model_config.sha256 == MODEL_CONFIG_SHA256
        and weights.path != model_config.path
        and weights.path.is_relative_to(ROOT / "edits" / "models")
        and model_config.path.is_relative_to(ROOT / "edits" / "models"),
        "Unpinned separation model",
    )
    boundaries = record(data["boundaries"])
    retained._require(set(boundaries) == BOUNDARIES, "Missing producer boundary evidence")
    terminal = record(data["terminal"])
    retained._require(
        set(terminal) == {"status", "log", "phases", "phase"}, "Invalid original terminal evidence"
    )
    phase = terminal["phase"]
    retained._require(
        isinstance(phase, str) and 0 < len(phase) <= 256 and "\0" not in phase,
        "Invalid original phase",
    )
    clock = record(data["clockStage"])
    retained._require(
        set(clock) == {"workflow", "receipt", "stageName", "inputs"},
        "Missing genuine clock stage binding",
    )
    workflow, receipt = _pin(clock["workflow"]), _pin(clock["receipt"])
    name = clock["stageName"]
    retained._require(
        isinstance(name, str) and receipt.path.name == f"{name}.json", "Clock stage name differs"
    )
    matches = tuple(
        stage for stage in parse_stages(retained._json(workflow.path)) if stage.name == name
    )
    retained._require(len(matches) == 1, "Original clock stage missing")
    stage = matches[0]
    inputs = _pins(clock["inputs"], maximum=retained.MAX_INPUTS)
    retained._require(
        {pin.path for pin in inputs} == set(stage.inputs)
        and roles["source"].path in stage.inputs
        and roles["audioClock"].path in stage.outputs
        and bool(stage.verify_command),
        "Clock stage must bind original source and verified clock",
    )
    timeout = data["timeoutSeconds"]
    retained._require(type(timeout) is int and 1 <= timeout <= 86400, "Invalid adoption deadline")
    return Job(
        retained.Pin(
            config, hashlib.sha256(retained._bytes(config, retained.MAX_METADATA_BYTES)).hexdigest()
        ),
        roles,
        {key: retained._tool(value) for key, value in tools.items()},
        (weights, model_config),
        _pins(data["providerInputs"], maximum=32),
        {key: _boundary(value) for key, value in boundaries.items()},
        {key: _pin(terminal[key]) for key in ("status", "log", "phases")},
        str(phase),
        workflow,
        receipt,
        inputs,
        stage,
        int(str(timeout)),
        _pins(data["supportingEvidence"], maximum=32) if "supportingEvidence" in data else (),
    )


def _expected(job: Job) -> tuple[retained.Pin, ...]:
    clock_result = retained._json(job.clock_receipt.path)
    retained._require(clock_result.get("status") == "complete", "Clock stage is not complete")
    outputs = record(clock_result.get("outputs"))
    retained._require(
        set(outputs) == {str(path) for path in job.clock_stage.outputs},
        "Clock receipt output membership differs",
    )
    clock_outputs = tuple(_pin({"path": path, "sha256": sha}) for path, sha in outputs.items())
    entries = (
        job.config,
        *job.roles.values(),
        *job.model,
        *job.provider,
        *job.supporting,
        *(boundary.evidence for boundary in job.boundaries.values()),
        *job.terminal.values(),
        job.clock_workflow,
        job.clock_receipt,
        *job.clock_inputs,
        *clock_outputs,
        *(tool.pin for tool in job.tools.values()),
        *retained._code(("replay_separation_adoption", "workflow")),
    )
    pins: dict[Path, str] = {}
    for pin in entries:
        retained._require(
            pin.path not in pins or pins[pin.path] == pin.sha256, "Conflicting input pins"
        )
        pins[pin.path] = pin.sha256
    retained._require(len(pins) <= retained.MAX_INPUTS, "Adoption input budget exceeded")
    return tuple(retained.Pin(path, sha) for path, sha in sorted(pins.items()))


def _argv(job: Job, action: str, plan: Path, receipt: Path, *, timed: bool) -> list[str]:
    work = plan.parent
    prefix = [
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
    ]
    command = ["replay.replay_separation_adoption", action, str(plan), str(receipt)]
    if timed:
        return [
            *prefix,
            "replay.phase_timer",
            str(work / "phases.jsonl"),
            f"separation-adoption-{action}",
            "--",
            str(job.tools["python"].invocation),
            "-B",
            "-m",
            *command,
        ]
    return [*prefix, *command]


def prepare(config: Path, work: Path) -> retained.Prepared:
    """Plan only: stat inputs and bounded metadata; no media/model hashing or decode."""
    _path(str(work))
    retained._require(
        work.parent.is_dir() and not work.exists(), "Use a new reconciliation work root"
    )
    job = _job(config)
    pins = _expected(job)
    plan, receipt = work / "plan.json", work / "adoption-evidence.json"
    outputs = {job.roles[name].path for name in ("vocals", "result")}
    direct = [str(pin.path) for pin in pins if retained._identity(pin.path)["bytes"] > 0]
    graphs: list[dict[str, object]] = []
    for action, name, adopting in (
        ("seal", "seal-retained-separation", False),
        ("verify", "adopt-retained-separation", True),
    ):
        graphs.append(
            {
                "version": 1,
                "stages": [
                    {
                        "name": name,
                        "command": _argv(job, action, plan, receipt, timed=True),
                        "verify_command": _argv(job, "verify", plan, receipt, timed=False),
                        "inputs": [
                            *(path for path in direct if not adopting or Path(path) not in outputs),
                            str(plan),
                            *([str(receipt)] if adopting else []),
                        ],
                        "outputs": sorted(str(path) for path in outputs)
                        if adopting
                        else [str(receipt)],
                        "timeout_seconds": job.timeout,
                        "adopt_existing": adopting,
                    }
                ],
            }
        )
    work.mkdir(mode=0o700)
    for name in ("scratch", "cache"):
        (work / name).mkdir(mode=0o700)
    retained._write(
        plan,
        {
            "version": 1,
            "protocol": PROTOCOL,
            "config": job.config.data(),
            "inputs": [pin.data() for pin in pins],
        },
    )
    result = retained.Prepared(
        work / "evidence-workflow.json", work / "adoption-workflow.json", work / "sequence.json"
    )
    for path, graph in zip(
        (result.evidence_workflow, result.adoption_workflow), graphs, strict=True
    ):
        retained._write(path, graph)
    retained._write(
        result.sequence,
        {
            "version": 1,
            "protocol": PROTOCOL,
            "order": [
                {
                    "workflow": str(result.evidence_workflow),
                    "state": str(work / "evidence-state"),
                    "requiredExit": 0,
                },
                {
                    "workflow": str(result.adoption_workflow),
                    "state": str(work / "adoption-state"),
                    "requiredExit": 0,
                },
            ],
            "originalGenerationReissued": False,
            "deliveryComplete": False,
        },
    )
    return result


def _load(plan: Path) -> tuple[Job, tuple[retained.Pin, ...]]:
    _path(str(plan))
    data = retained._json(plan)
    retained._require(
        set(data) == {"version", "protocol", "config", "inputs"}
        and type(data["version"]) is int
        and data["version"] == 1
        and data["protocol"] == PROTOCOL,
        "Invalid adoption plan",
    )
    config = _pin(data["config"])
    retained._require(digest(config.path) == config.sha256, "Configuration pin changed")
    job = _job(config.path)
    pins = _pins(data["inputs"], maximum=retained.MAX_INPUTS)
    retained._require(pins == _expected(job), "Input membership or code pins changed")
    return job, pins


def _snapshot(pin: retained.Pin) -> dict[str, str]:
    """Read existing JSON inventories or strict sha256_check text, never execute code."""
    data = retained._bytes(pin.path, retained.MAX_METADATA_BYTES)
    if data.lstrip().startswith(b"{"):
        raw = record(record(json.loads(data, object_pairs_hook=retained._pairs)).get("inputs"))
        values = list(raw.items())
    else:
        retained._require(data.endswith(b"\n"), "Incomplete producer input inventory")
        values = []
        for line in data.decode("utf-8").splitlines():
            sha_text, separator, path_text = line.partition("  ")
            retained._require(bool(separator), "Invalid producer input inventory")
            values.append((path_text, sha_text))
    result: dict[str, str] = {}
    for path, sha in values:
        retained._require(
            isinstance(path, str) and bool(path) and "\0" not in path, "Invalid inventory path"
        )
        name = Path(str(path))
        retained._require(
            name.is_absolute()
            and str(name) == path
            and ".." not in name.parts
            and path not in result,
            "Invalid inventory reference",
        )
        # Preserve historical HF alias names instead of rewriting old inventories.
        # Their actual targets and aliases are checked at both operation boundaries.
        _pin({"path": str(name.resolve(strict=True)), "sha256": sha}, local=False)
        result[str(path)] = str(sha)
    retained._require(
        0 < len(result) <= retained.MAX_INPUTS,
        "Invalid snapshot membership",
    )
    return result


def _phase(job: Job) -> dict[str, object]:
    retained._require(
        retained._bytes(job.terminal["status"].path, 64) == b"0\n",
        "Original separation is not successful",
    )
    data = retained._bytes(job.terminal["phases"].path, retained.MAX_PHASE_BYTES)
    retained._require(data.endswith(b"\n"), "Original phase log is incomplete")
    rows = [
        record(json.loads(line, object_pairs_hook=retained._pairs)) for line in data.splitlines()
    ]
    matches = [row for row in rows if row.get("phase") == job.phase]
    retained._require(len(matches) == 1, "Original separation phase must be unique")
    row = matches[0]
    retained._require(
        set(row) == {"phase", "start", "end", "seconds", "exit"}
        and type(row["exit"]) is int
        and row["exit"] == 0,
        "Original separation phase failed",
    )
    times = [row[key] for key in ("start", "end", "seconds")]
    retained._require(
        all(type(value) in (int, float) and math.isfinite(float(str(value))) for value in times),
        "Invalid original phase timing",
    )
    start, end, seconds = (float(str(value)) for value in times)
    retained._require(0 < start <= end and 0 <= seconds <= 86400, "Invalid original phase duration")
    return row


def _domain(job: Job) -> dict[str, object]:
    phase = _phase(job)
    expected = {
        "code": {str(pin.path): pin.sha256 for pin in retained._code(PRODUCER_MODULES)},
        "model": {str(pin.path): pin.sha256 for pin in job.model},
        "provider": {str(pin.path): pin.sha256 for pin in job.provider},
    }
    boundary_results: dict[str, object] = {}
    for role, bindings in expected.items():
        for suffix in ("Before", "Terminal"):
            name = f"{role}{suffix}"
            boundary = job.boundaries[name]
            references: dict[str, str] = {}
            if boundary.inventory:
                observed = _snapshot(boundary.evidence)
                references = {path: str(Path(path).resolve(strict=True)) for path in observed}
                resolved = {references[path]: sha for path, sha in observed.items()}
                allowed = {
                    **bindings,
                    str(job.roles["source"].path): job.roles["source"].sha256,
                }
                retained._require(
                    len(resolved) == len(observed)
                    and all(resolved.get(path) == sha for path, sha in bindings.items())
                    and all(allowed.get(path) == sha for path, sha in resolved.items()),
                    f"Retained {role} boundaries changed",
                )
            else:
                # Existing descriptive metadata is retained as evidence, not turned
                # into a generation-time byte inventory or execution attestation.
                retained._json(boundary.evidence.path)
            boundary_results[name] = {
                "evidence": boundary.evidence.data(),
                "inputInventoryVerified": boundary.inventory,
                "historicalCaptureBoundaryVerified": False,
                "resolvedReferences": references,
            }
    clock_receipt = retained._json(job.clock_receipt.path)
    retained._require(
        clock_receipt
        == {
            "status": "complete",
            "fingerprint": _fingerprint(job.clock_stage),
            "outputs": {str(path): digest(path) for path in job.clock_stage.outputs},
        },
        "Genuine clock stage inputs or outputs changed",
    )
    result = retained._json(job.roles["result"].path)
    prepared = result.get("preparedInput")
    retained._require(
        type(prepared) is bool
        and result.get("device") == "cpu"
        and result.get("loudnessNormalization") is False
        and result.get("timestampCorrection") is False
        and result.get("deliveryComplete") is False,
        "Separation policy differs",
    )
    roles = {name: pin.path for name, pin in job.roles.items()}
    source = roles["modelInput"] if prepared else roles["source"]
    retained._require(
        roles["result"].name == "result.json"
        and roles["vocals"] == roles["result"].parent / "vocals.wav"
        and result.get("modelInput") == str(roles["modelInput"])
        and result.get("vocals") == str(roles["vocals"])
        and result.get("source") == str(source),
        "Separation input/output binding differs",
    )
    gain = result.get("gainDB")
    retained._require(
        type(gain) in (int, float)
        and math.isfinite(float(str(gain)))
        and float(str(gain)) <= 0
        and (not prepared or gain == 0),
        "Separation gain policy differs",
    )
    # This is the exact readonly implementation used by delivery_audio --verify.
    verify_material(source, roles["result"].parent)
    with wave.open(str(roles["modelInput"]), "rb") as stream:
        samples = stream.getnframes()
    audio = AudioFrameClock.from_probe(retained._json(roles["audioClock"], budget=MAX_CLOCK_BYTES))
    retained._require(
        abs(Fraction(samples, 44100) - Fraction(audio.sample_ends[-1], audio.sample_rate))
        <= Fraction(1, 44100),
        "Separated PCM differs from native decoded audio duration",
    )
    return {
        "originalPhase": phase,
        "currentProducerPins": expected,
        "retainedBoundaryEvidence": boundary_results,
        "clockStage": {
            "workflow": job.clock_workflow.data(),
            "receipt": job.clock_receipt.data(),
            "stageName": job.clock_stage.name,
        },
        "nativeAudioClock": {
            "sampleRate": audio.sample_rate,
            "timeBase": str(audio.time_base),
            "firstPTS": audio.pts[0],
            "lastPTS": audio.pts[-1],
            "decodedSamples": audio.sample_ends[-1],
        },
        "modelInputSamples": samples,
        "vocalsSamples": samples,
        "sampleRate": 44100,
        "channels": 2,
        "preparedInput": prepared,
        "device": "cpu",
    }


def _verified(plan: Path) -> dict[str, object]:
    job, pins = _load(plan)
    bindings = (*pins, retained.Pin(plan, digest(plan)))
    before = retained._fresh(bindings)
    domains = _domain(job)
    retained._require(
        retained._fresh(bindings) == before, "Inputs changed during separation verification"
    )
    for value in record(domains["retainedBoundaryEvidence"]).values():
        references = record(record(value)["resolvedReferences"])
        retained._require(
            all(
                str(Path(name).resolve(strict=True)) == target
                for name, target in references.items()
            ),
            "Producer inventory alias changed",
        )
    retained._require(
        all(tool.invocation.resolve(strict=True) == tool.pin.path for tool in job.tools.values()),
        "Tool alias changed",
    )
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "kind": "limited-current-input-retained-outcome",
        "inputs": {str(pin.path): pin.sha256 for pin in bindings},
        "outputs": {
            str(job.roles[name].path): job.roles[name].sha256 for name in ("vocals", "result")
        },
        "source": job.roles["source"].data(),
        "domains": domains,
        "currentReaderCode": {
            str(pin.path): pin.sha256 for pin in retained._code(("replay_separation_adoption",))
        },
        "currentInputBytesVerified": True,
        "currentPCMClockVerified": True,
        "currentMaterialDomainVerified": True,
        "historicalGenerationBindingFullyVerified": False,
        "historicalSourceDigestAtDispatchVerified": False,
        "historicalOriginalArgvRecorded": False,
        "historicalToolPinsAtDispatchVerified": False,
        "historicalInitialCodeSnapshotVerified": False,
        "historicalSourceToPCMContentVerified": False,
        "originalGenerationReceiptReissued": False,
        "speechOrMusicQualityApproved": False,
        "synchronizationApproved": False,
        "candidateOnly": True,
        "deliveryComplete": False,
    }


def seal(plan: Path, receipt: Path) -> None:
    """Create new independent evidence after true-byte endpoint and PCM checks."""
    _path(str(receipt))
    retained._require(
        receipt.parent == plan.parent and not receipt.exists(),
        "Use a new evidence receipt in the plan work root",
    )
    retained._write(receipt, _verified(plan))


def verify(plan: Path, receipt: Path) -> None:
    """Read-only check; original outcomes, phases and workflow records never change."""
    _path(str(receipt))
    retained._require(receipt.parent == plan.parent, "Evidence receipt escapes plan work root")
    before = retained._bytes(receipt, retained.MAX_METADATA_BYTES)
    retained._require(retained._json(receipt) == _verified(plan), "Adoption evidence changed")
    retained._require(
        retained._bytes(receipt, retained.MAX_METADATA_BYTES) == before,
        "Evidence changed during verification",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "seal", "verify"))
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.first, args.second)
            print(result.sequence)
        elif args.action == "seal":
            seal(args.first, args.second)
        else:
            verify(args.first, args.second)
    except (ReplayError, StageError, OSError, ValueError, SyntaxError, wave.Error) as error:
        print(f"SEPARATION_ADOPTION_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
