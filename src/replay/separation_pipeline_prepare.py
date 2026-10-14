"""Seal clocked model PCM and prepare the existing two-worker pipeline plan.

This bridge never infers or adopts a historical separation output. Explicit
``adopt`` verifies an already completed, single extraction; it never extracts
again. Source presentation timing remains a separate downstream projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from replay.audio_clock import AudioFrameClock
from replay.bgm_separation import MODEL_SAMPLE_RATE
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.delivery_audio import input_command
from replay.input_hash_snapshot import InputHashSnapshot
from replay.native_invocation import canonical, durable, identity, pin, read, require
from replay.recordings import record
from replay.separation_window_pipeline import context, load_plan, pcm_info
from replay.stages import StageError

GAIN_DB = -0.1
BLOCK_SAMPLES = 65536
MAX_CLOCK_BYTES = 128 * 1024 * 1024
EXTRACTION_TIMEOUT = 7200
MAX_HASH_OUTPUT_BYTES = 256
MAX_HASH_ERROR_BYTES = 8192
CONFIG_FIELDS = frozenset(
    {
        "version",
        "kind",
        "source",
        "mediaRoot",
        "preparation",
        "audioClock",
        "audioClockProducer",
        "ffmpeg",
        "directory",
        "gainDB",
        "codeInputs",
        "completedExtraction",
    }
)
ROLES = ("source", "preparation", "audioClock", "audioClockProducer", "ffmpeg")


@dataclass(frozen=True)
class Preparation:
    path: Path
    directory: Path
    references: dict[str, tuple[Path, str]]
    code: dict[Path, str]
    completion: dict[str, object] | None

    @property
    def pcm(self) -> Path:
        return self.directory / "model-input.wav"

    @property
    def receipt(self) -> Path:
        return self.directory / "pcm-preparation.json"


class Arguments(argparse.Namespace):
    action: str
    config: Path
    template: Path
    output: Path
    verify: bool


def code_inputs() -> dict[Path, str]:
    return {
        item.path: item.sha256
        for item in capture(("separation_pipeline_prepare", "code_dependency_inventory"))
    }


def _reference(value: object) -> tuple[Path, str]:
    row = record(value)
    path = canonical(row.get("path"))
    digest = row.get("sha256")
    require(
        set(row) == {"path", "sha256"}
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "Expected an exact path/SHA256 reference",
    )
    identity(path)
    return path, str(digest)


def load_config(path: Path) -> Preparation:
    path = canonical(str(path))
    root = Path.cwd().resolve()
    require(path.is_relative_to(root / "edits/work"), "Preparation config must remain local")
    data = read(path)
    require(
        set(data) == CONFIG_FIELDS
        and type(data.get("version")) is int
        and data.get("version") == 1
        and data.get("kind") == "separation-pcm-preparation"
        and type(data.get("gainDB")) is float
        and data.get("gainDB") == GAIN_DB,
        "Unsupported PCM preparation contract",
    )
    refs = {role: _reference(data.get(role)) for role in ROLES}
    media_root = canonical(data.get("mediaRoot"))
    require(media_root.is_dir(), "Expected explicit authorized media root")
    require(refs["source"][0].is_relative_to(media_root), "Source escapes its media root")
    require(
        all(refs[role][0].is_relative_to(root) for role in ROLES[1:-1]),
        "Clock/preparation evidence must remain local",
    )
    directory = canonical(data.get("directory"))
    require(
        directory.is_relative_to(root / "edits/work") and directory.is_dir(),
        "PCM directory must be an existing local work directory",
    )
    values = data.get("codeInputs")
    if not isinstance(values, list) or not values:
        raise ReplayError("Missing preparation AST pins")
    pairs = tuple(_reference(value) for value in values)
    code = dict(pairs)
    require(len(code) == len(pairs) and code == code_inputs(), "Preparation AST closure differs")
    require(
        len({item[0] for item in refs.values()}) == len(refs)
        and not set(code).intersection(item[0] for item in refs.values())
        and directory / "model-input.wav" not in {item[0] for item in refs.values()},
        "Overlapping preparation input roles",
    )
    completion_value = data.get("completedExtraction")
    completion = None if completion_value is None else record(completion_value)
    if completion is not None:
        require(set(completion) == {"status", "phases", "phase"}, "Invalid completion evidence")
        require(
            isinstance(completion["phase"], str) and bool(completion["phase"]),
            "Missing completed phase name",
        )
        refs["completedStatus"] = _reference(completion["status"])
        refs["completedPhases"] = _reference(completion["phases"])
        require(
            all(
                refs[role][0].is_relative_to(directory)
                for role in ("completedStatus", "completedPhases")
            ),
            "Extraction completion evidence escapes output namespace",
        )
    return Preparation(path, directory, refs, code, completion)


def _snapshot(config: Preparation) -> InputHashSnapshot:
    return InputHashSnapshot.capture(
        (
            *config.references.values(),
            *config.code.items(),
            (config.path, str(pin(config.path)["sha256"])),
        )
    )


def _clock(config: Preparation) -> AudioFrameClock:
    path, digest = config.references["audioClock"]
    require(identity(path)["bytes"] <= MAX_CLOCK_BYTES, "Audio clock exceeds its budget")
    producer = read(config.references["audioClockProducer"][0])
    source, source_digest = config.references["source"]
    require(
        producer.get("version") == 1
        and producer.get("protocol") == "replay-preparation-v1"
        and producer.get("kind") == "audio"
        and producer.get("sharedProducerCompleted") is True
        and producer.get("candidateOnly") is True
        and producer.get("deliveryComplete") is False
        and producer.get("preparationSHA256") == config.references["preparation"][1]
        and producer.get("source") == {"path": str(source), "sha256": source_digest}
        and producer.get("outputs") == {str(path): digest},
        "Completed audio producer does not bind this source and clock",
    )
    # Audio frame evidence is larger than the generic invocation JSON limit.
    return AudioFrameClock.from_probe(json.loads(path.read_bytes()))


def _completed(config: Preparation) -> dict[str, object]:
    command = input_command(
        config.references["source"][0],
        config.pcm,
        gain=GAIN_DB,
        ffmpeg=str(config.references["ffmpeg"][0]),
    )
    if config.completion is None:
        complete = read(config.directory / "extraction-complete.json")
        require(
            complete == {"version": 1, "config": pin(config.path), "command": command, "exit": 0},
            "Shared extraction has no unchanged completed invocation",
        )
        return {
            "origin": "shared-extract",
            "complete": pin(config.directory / "extraction-complete.json"),
        }
    require(
        config.references["completedStatus"][0].read_text(encoding="utf-8") == "0\n",
        "External extraction is not terminal zero",
    )
    phases = config.references["completedPhases"][0]
    require(identity(phases)["bytes"] <= MAX_CLOCK_BYTES, "Phase evidence exceeds budget")
    records = tuple(
        record(json.loads(line)) for line in phases.read_text(encoding="utf-8").splitlines()
    )
    selected = tuple(row for row in records if row.get("phase") == config.completion["phase"])
    require(
        len(selected) == 1
        and type(selected[0].get("exit")) is int
        and selected[0].get("exit") == 0,
        "Expected exactly one completed successful extraction phase",
    )
    return {"origin": "explicit-completed-extraction-adoption", "evidence": config.completion}


def _pcm_detail(config: Preparation) -> dict[str, object]:
    clock = _clock(config)
    samples = pcm_info(config.pcm)
    require(
        abs(samples * clock.sample_rate - clock.sample_ends[-1] * MODEL_SAMPLE_RATE)
        <= clock.sample_rate,
        "Prepared PCM differs from decoded source duration by more than one output sample",
    )
    before = identity(config.pcm)
    decoded_bytes = 0
    payload = hashlib.sha256()
    with config.pcm.open("rb") as stream, wave.open(stream, "rb") as audio:
        while block := audio.readframes(BLOCK_SAMPLES):
            decoded_bytes += len(block)
            payload.update(block)
    require(
        decoded_bytes == samples * 4 and identity(config.pcm) == before,
        "Prepared PCM is truncated or changed",
    )
    expected_digest = expected_pcm_digest(config)
    require(
        payload.hexdigest() == expected_digest,
        "Prepared PCM payload differs from the pinned source/filter/gain",
    )
    return {
        "inputClock": "continuous-decoded-pcm",
        "sourceSampleRate": clock.sample_rate,
        "sourceDecodedSamples": clock.sample_ends[-1],
        "sampleRate": MODEL_SAMPLE_RATE,
        "channels": 2,
        "sampleWidthBytes": 2,
        "inputSampleCount": samples,
        "outputSampleCount": samples,
        "sourceProjectionApplications": 0,
        "timestampCorrection": False,
        "loudnessNormalization": False,
        "gainDB": GAIN_DB,
        "gainPolicy": "explicit fixed preparation gain; no peak measurement claimed",
        "pcmPayloadSHA256": payload.hexdigest(),
        "expectedFFmpegPCM16SHA256": expected_digest,
        "currentSourceContentEquivalent": True,
        "qualityApproved": False,
        "productionBackendAdopted": False,
        "deliveryComplete": False,
    }


def expected_pcm_digest(config: Preparation) -> str:
    """Decode to the hash muxer, without creating a WAV or running inference."""
    command = input_command(
        config.references["source"][0],
        config.pcm,
        gain=GAIN_DB,
        ffmpeg=str(config.references["ffmpeg"][0]),
    )[:-1]
    command.extend(("-f", "hash", "-hash", "sha256", "-"))
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        timeout=EXTRACTION_TIMEOUT,
    )
    require(
        len(result.stdout) <= MAX_HASH_OUTPUT_BYTES and len(result.stderr) <= MAX_HASH_ERROR_BYTES,
        "FFmpeg PCM digest response exceeds its budget",
    )
    value = result.stdout.decode("ascii").strip()
    require(value.startswith("SHA256="), "Missing FFmpeg PCM payload SHA256")
    digest = value.removeprefix("SHA256=")
    require(
        len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
        "Invalid FFmpeg PCM payload SHA256",
    )
    return digest


def seal(config_path: Path) -> Path:
    """Verify one completed extraction and seal it without any model invocation."""
    config = load_config(config_path)
    require(not config.receipt.exists() and not config.receipt.is_symlink(), "PCM receipt occupied")
    snapshot = _snapshot(config)
    completed = _completed(config)
    detail = _pcm_detail(config)
    output = pin(config.pcm)
    inputs = {str(path): digest for path, digest in snapshot.finish().items()}
    require(
        pin(config.pcm) == output and _completed(config) == completed,
        "Prepared PCM/completion changed before sealing",
    )
    durable(
        config.receipt,
        {
            "version": 1,
            "kind": "verified-separation-pcm-preparation",
            "inputs": inputs,
            "output": output,
            "expectedExtractionCommand": input_command(
                config.references["source"][0],
                config.pcm,
                gain=GAIN_DB,
                ffmpeg=str(config.references["ffmpeg"][0]),
            ),
            "completedExtraction": completed,
            "externalExecutedArgvAttested": False,
            "externalPreExtractionSnapshotAttested": False,
            **detail,
        },
    )
    return config.receipt


def verify(config_path: Path) -> dict[str, object]:
    """Recheck the complete PCM receipt and source/tool/code pins, without writes."""
    config = load_config(config_path)
    before = pin(config.receipt)
    receipt = read(config.receipt)
    snapshot = _snapshot(config)
    completed = _completed(config)
    detail = _pcm_detail(config)
    output = pin(config.pcm)
    inputs = {str(path): digest for path, digest in snapshot.finish().items()}
    expected = {
        "version": 1,
        "kind": "verified-separation-pcm-preparation",
        "inputs": inputs,
        "output": output,
        "expectedExtractionCommand": input_command(
            config.references["source"][0],
            config.pcm,
            gain=GAIN_DB,
            ffmpeg=str(config.references["ffmpeg"][0]),
        ),
        "completedExtraction": completed,
        "externalExecutedArgvAttested": False,
        "externalPreExtractionSnapshotAttested": False,
        **detail,
    }
    require(receipt == expected and pin(config.receipt) == before, "PCM preparation changed")
    require(
        pin(config.pcm) == output and _completed(config) == completed,
        "PCM/completion changed during verification",
    )
    return receipt


def extract(config_path: Path) -> Path:
    """Run the original FFmpeg PCM command once; retain uncertain/failed outputs."""
    config = load_config(config_path)
    attempt, log = config.directory / "extraction-attempt.json", config.directory / "extract.stderr"
    require(
        config.completion is None
        and not any(
            path.exists() or path.is_symlink()
            for path in (
                config.pcm,
                config.receipt,
                attempt,
                log,
                config.directory / "extraction-complete.json",
            )
        ),
        "Extraction already attempted; reconcile instead of repeating it",
    )
    snapshot = _snapshot(config)
    _clock(config)
    command = input_command(
        config.references["source"][0],
        config.pcm,
        gain=GAIN_DB,
        ffmpeg=str(config.references["ffmpeg"][0]),
    )
    durable(attempt, {"version": 1, "command": command, "status": "reserved"})
    with log.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stream,
            check=True,
            timeout=EXTRACTION_TIMEOUT,
        )
    snapshot.finish()
    config.pcm.chmod(0o600)
    durable(
        config.directory / "extraction-complete.json",
        {"version": 1, "config": pin(config.path), "command": command, "exit": 0},
    )
    return seal(config.path)


def prepare_plan(
    config_path: Path, template: Path, output: Path, *, verify_only: bool = False
) -> Path:
    """Fill only the measured PCM SHA in an explicit existing pipeline template."""
    config = load_config(config_path)
    template, output = canonical(str(template)), canonical(str(output))
    require(
        template.is_relative_to(Path.cwd().resolve() / "edits/work")
        and output.parent == config.directory
        and output not in (template, config.path, config.receipt, config.pcm),
        "Pipeline plan must have a new local, distinct path",
    )
    receipt = verify(config.path)
    template_pin, receipt_pin = pin(template), pin(config.receipt)
    definition = read(template)
    require(
        definition.get("source") == {"path": str(config.pcm), "sha256": None},
        "Template must declare this prepared PCM with unresolved SHA only",
    )
    expected = {
        **definition,
        "source": {"path": str(config.pcm), "sha256": record(receipt["output"])["sha256"]},
    }
    if verify_only:
        require(read(output) == expected, "Prepared pipeline plan changed")
    else:
        require(not output.exists() and not output.is_symlink(), "Pipeline plan occupied")
        durable(output, expected)
    plan = load_plan(output)
    bound = context(plan)
    require(
        pin(template) == template_pin
        and pin(config.receipt) == receipt_pin
        and context(plan) == bound,
        "Pipeline plan preparation inputs changed",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("extract", "adopt", "verify"):
        command = commands.add_parser(action)
        command.add_argument("config", type=Path)
    prepare = commands.add_parser("prepare-plan")
    prepare.add_argument("config", type=Path)
    prepare.add_argument("template", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv, namespace=Arguments())
    try:
        if args.action == "prepare-plan":
            result: object = prepare_plan(
                args.config, args.template, args.output, verify_only=args.verify
            )
        else:
            actions = {"extract": extract, "adopt": seal, "verify": verify}
            result = actions[args.action](args.config)
        print(result)
    except (OSError, ValueError, ReplayError, StageError, subprocess.SubprocessError) as error:
        print(f"SEPARATION_PREPARATION_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
