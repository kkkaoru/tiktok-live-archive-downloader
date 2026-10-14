"""Derive current quiet cuts from authenticated retained peak measurements."""

import argparse
import hashlib
import json
import re
import sys
import wave
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path

from replay import asr_clock_projection as projection
from replay import caption_continuation_verify as verifier
from replay import cleanup_lane_successor as leases
from replay import code_dependency_inventory as inventory
from replay import continuous_quiet_plan as quiet
from replay.audio_clock import AudioFrameClock
from replay.core import ReplayError
from replay.delivery_media import read_clock
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.quiet_cuts import quiet_cut_frames
from replay.recordings import record
from replay.sequence_journal import bounded_read, decode, encode
from replay.stages import Stage, file_digest
from replay.waveform_quiet import measured_quiet_runs
from replay.workflow import load_stages

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
KIND = "retained-quiet-peaks-request"
SHA256 = re.compile(r"[a-f0-9]{64}")
BASELINE_ROLES = frozenset({"config", "cuts", "peaks", "workflow", "checkpoint", "readerClosure"})
ARTIFACT_ROLES = frozenset(
    {"labels", "stereoPCM", "audioClock", "videoClock", "projectionConfig", "lineageEvidence"}
)
MUTABLE_ROLES = frozenset({"labels", "projectionConfig", "lineageEvidence"})
PURE_MODULES = (
    "continuous_quiet_plan",
    "waveform_quiet",
    "quiet_cuts",
    "audio_clock",
    "caption_projection",
    "caption_repair",
    "subtitle_clock",
    "delivery_media",
    "caption_continuation_verify",
    "asr_clock_projection",
)


@dataclass(frozen=True)
class Job:
    request: Path
    baseline: dict[str, Path]
    references: dict[str, object]
    current: Path
    output: Path
    pins: dict[Path, str]


def _path(value: object, *, local: bool = True) -> Path:
    if not isinstance(value, str):
        raise ReplayError("Expected an explicit canonical retained input path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ReplayError("Retained inputs must use canonical absolute paths")
    if local and not path.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("Retained metadata must remain checkout-local")
    if any(path.is_relative_to(ROOT / name) for name in ("private", "downloads")):
        raise ReplayError("Private and download inputs are not accepted")
    return path


def _sha(value: object) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ReplayError("Expected an exact lowercase SHA256")
    return value


def _read(path: Path) -> dict[str, object]:
    return decode(bounded_read(path))


def _pin(pins: dict[Path, str], path: Path, digest: str) -> None:
    if path in pins and pins[path] != digest:
        raise ReplayError("Conflicting retained input bindings")
    pins[path] = digest


def _reference(value: object, pins: dict[Path, str], *, local: bool = True) -> Path:
    reference = record(value)
    if set(reference) != {"path", "sha256"}:
        raise ReplayError("Expected an exact retained artifact reference")
    path = _path(reference["path"], local=local)
    _pin(pins, path, _sha(reference["sha256"]))
    return path


def _load(request: Path, sha256: str | None) -> Job:
    _path(str(request))
    digest = file_digest(request)
    if sha256 is not None and digest != _sha(sha256):
        raise ReplayError("Retained request pin changed")
    raw = _read(request)
    if set(raw) != {"version", "kind", "baseline", "currentConfig", "output"} or (
        type(raw["version"]) is not int or raw["version"] != 1 or raw["kind"] != KIND
    ):
        raise ReplayError("Invalid retained quiet request")
    baseline = record(raw["baseline"])
    if set(baseline) != BASELINE_ROLES:
        raise ReplayError("Incomplete original quiet measurement evidence")
    pins = {request: digest}
    paths = {name: _reference(value, pins) for name, value in baseline.items()}
    current = _reference(raw["currentConfig"], pins)
    output = _path(raw["output"])
    if (
        paths["peaks"].name != "peaks.txt"
        or paths["cuts"].parent != paths["peaks"].parent
        or output.parent != paths["peaks"].parent
        or output.suffix != ".json"
        or output in {*pins, *paths.values()}
        or output.is_symlink()
    ):
        raise ReplayError("Retained cuts require a new adjacent publication leaf")
    return Job(request, paths, baseline, current, output, pins)


def _source(path: Path) -> bool:
    return path.parent == PACKAGE and path.suffix == ".py" and path == path.resolve()


def _historical_fingerprint(stage: Stage, inputs: Mapping[str, str]) -> str:
    definition = {
        "command": stage.command,
        "verifyCommand": stage.verify_command,
        "inputs": dict(inputs),
        "outputs": [str(path) for path in stage.outputs],
        "timeoutSeconds": stage.timeout_seconds,
        "workingDirectory": str(Path.cwd().resolve()),
    }
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode("utf-8")).hexdigest()


def _stage(job: Job) -> tuple[Stage, dict[str, str]]:
    stages = [
        stage
        for stage in load_stages(job.baseline["workflow"])
        if set(stage.outputs) == {job.baseline["cuts"], job.baseline["peaks"]}
    ]
    if len(stages) != 1:
        raise ReplayError("Require exactly one original quiet measurement Stage")
    stage = stages[0]
    command = stage.command
    if (
        len(command) != 8
        or command[:4] != (sys.executable, "-B", "-m", "replay.continuous_quiet_plan")
        or command[4:7]
        != (str(job.baseline["config"]), str(job.baseline["cuts"].parent), "--ffmpeg")
        or stage.verify_command
        != (
            sys.executable,
            "-B",
            "-m",
            "replay.caption_continuation_verify",
            "quiet",
            str(job.baseline["config"]),
            str(job.baseline["cuts"]),
        )
        or job.baseline["readerClosure"] not in stage.inputs
        or job.baseline["config"] not in stage.inputs
        or job.baseline["checkpoint"].name != stage.name + ".json"
        or stage.adopt_existing
    ):
        raise ReplayError("Original quiet Stage command or verifier differs")
    closure = _read(job.baseline["readerClosure"])
    if set(closure) != {"inputs"}:
        raise ReplayError("Expected original source-only reader inventory")
    sources: dict[str, str] = {}
    for name, value in record(closure["inputs"]).items():
        path = _path(name, local=False)
        if not _source(path):
            raise ReplayError("Historical inventory contains a non-source pin")
        sources[name] = _sha(value)
    for module in PURE_MODULES:
        path = PACKAGE / f"{module}.py"
        if sources.get(str(path)) != file_digest(path):
            raise ReplayError("Retained quiet algorithms changed; measurement cannot be adopted")
    return stage, sources


def _config(path: Path, pins: dict[Path, str]) -> tuple[dict[str, object], dict[str, Path]]:
    data = _read(path)
    fps = data.get("fps")
    artifacts = record(data.get("artifacts"))
    if (
        type(data.get("version")) is not int
        or data.get("version") != 1
        or data.get("pcmClock") != "continuous-decoded-pcm"
        or type(fps) is not int
        or not 1 <= fps <= 120
        or set(artifacts) != ARTIFACT_ROLES
    ):
        raise ReplayError("Invalid retained quiet configuration")
    paths = {name: _reference(value, pins) for name, value in artifacts.items()}
    projection = _read(paths["projectionConfig"])
    projected = record(projection["artifacts"])
    mono = _reference(projected["pcm"], pins)
    labels = _read(paths["labels"])
    lineage = _read(paths["lineageEvidence"])
    if (
        lineage.get("clock") != "continuous-decoded-pcm"
        or lineage.get("stereoPCM") != artifacts["stereoPCM"]
        or lineage.get("monoPCM") != {"path": str(mono), "sha256": pins[mono]}
        or lineage.get("audioClock") != artifacts["audioClock"]
        or lineage.get("transforms")
        != ["concatenate-decoded-frames", "duration-preserving-resample"]
        or projected.get("audioClock") != artifacts["audioClock"]
        or labels.get("configSHA256") != pins[paths["projectionConfig"]]
        or record(labels.get("parentSHA256")).get(str(mono)) != pins[mono]
    ):
        raise ReplayError("Retained quiet inputs do not preserve the same decoded PCM lineage")
    return data, paths


def _policy(old: dict[str, object], current: dict[str, object]) -> None:
    if {
        name: value for name, value in old.items() if name not in {"artifacts", "templateSHA256"}
    } != {
        name: value
        for name, value in current.items()
        if name not in {"artifacts", "templateSHA256"}
    } or any(
        record(old["artifacts"])[role] != record(current["artifacts"])[role]
        for role in ARTIFACT_ROLES - MUTABLE_ROLES
    ):
        raise ReplayError("Retained quiet source, clock or planning policy changed")


def _derive(
    job: Job, data: dict[str, object], paths: dict[str, Path], sources: dict[str, str]
) -> dict[str, object]:
    fps = data["fps"]
    if type(fps) is not int:
        raise ReplayError("Retained quiet frame rate must be integral")
    clock = AudioFrameClock.from_probe(projection._read(paths["audioClock"]))
    with wave.open(str(paths["stereoPCM"]), "rb") as stream:
        samples, rate = stream.getnframes(), stream.getframerate()
        if (
            stream.getnchannels() != 2
            or stream.getsampwidth() != 2
            or stream.getcomptype() != "NONE"
            or not 8000 <= rate <= 192000
            or not 0 < samples <= rate * 21600
            or abs(Fraction(samples, rate) - Fraction(clock.sample_ends[-1], clock.sample_rate))
            > Fraction(1, rate)
        ):
            raise ReplayError("Retained stereo PCM does not preserve decoded duration")
    with job.baseline["peaks"].open(encoding="utf-8") as lines:
        measured = measured_quiet_runs(
            lines, sample_count=samples, sample_rate=rate, window_samples=max(1, rate // 50)
        )
    mapped = quiet.source_quiet(measured, clock, measurement_rate=rate)
    video = read_clock(paths["videoClock"])
    frames = ceil((video.pts[-1] + video.last_duration) * video.time_base * fps)
    cuts = quiet_cut_frames(
        mapped, quiet._labels(_read(paths["labels"])), source_frames=frames, fps=fps
    )
    baseline = _read(job.baseline["cuts"])
    return {
        "version": 1,
        "clock": "source-presentation",
        "fps": fps,
        "sourceFrames": frames,
        "cuts": [asdict(cut) for cut in cuts],
        "sourceQuiet": [asdict(span) for span in mapped],
        "parentSHA256": {str(path): job.pins[path] for path in paths.values()},
        "configSHA256": job.pins[job.current],
        "peakSHA256": job.pins[job.baseline["peaks"]],
        "plannerSHA256": file_digest(Path(quiet.__file__)),
        "command": baseline["command"],
        "lineageIsCallerDeclared": True,
        "nonSpeechRemovalVerified": False,
        "humanReviewed": False,
        "deliveryComplete": False,
        "retainedDerivation": {
            "version": 1,
            "request": {"path": str(job.request), "sha256": job.pins[job.request]},
            "baseline": job.references,
            "helperSHA256": file_digest(Path(__file__)),
            "historicalSourcePins": sources,
            "historicalReaderEqualsCurrent": False,
            "waveformMeasurementRepeated": False,
            "qualityApproved": False,
        },
    }


def run(request: Path, *, verify: bool = False, sha256: str | None = None) -> Path:
    """Recompute protected cuts from genuine retained peaks, never remeasure audio."""
    job = _load(request, sha256)
    stage, historical_sources = _stage(job)
    old_data, old_paths = _config(job.baseline["config"], job.pins)
    data, paths = _config(job.current, job.pins)
    _policy(old_data, data)
    old_projection = _read(old_paths["projectionConfig"])
    current_projection = _read(paths["projectionConfig"])
    if record(old_projection["artifacts"])["pcm"] != record(current_projection["artifacts"])["pcm"]:
        raise ReplayError("Retained recognition mono PCM identity changed")
    for source in inventory.capture(("retained_quiet_peaks",)):
        _pin(job.pins, source.path, source.sha256)
    with ExitStack() as stack:
        state = job.baseline["checkpoint"].parent
        for path in (state / ".workflow.lock", state / f"{stage.name}.lock"):
            stack.enter_context(leases._unlocked(path))
        stage_material = tuple(path for path in stage.inputs if not _source(path))
        cache = InputHashCache((*job.pins, *stage_material), allow_uncached_overflow=True)
        stack.enter_context(input_hash_context(cache))
        cache.prime()
        if any(file_digest(path) != digest for path, digest in job.pins.items()):
            raise ReplayError("Retained measurement input bytes changed")
        for path in stage_material:
            _pin(job.pins, path, file_digest(path))
        inputs: dict[str, str] = {}
        for path in stage.inputs:
            if _source(path):
                if str(path) not in historical_sources:
                    raise ReplayError("Original Stage lacks a historical source binding")
                inputs[str(path)] = historical_sources[str(path)]
            else:
                inputs[str(path)] = job.pins[path]
        checkpoint = _read(job.baseline["checkpoint"])
        if checkpoint != {
            "status": "complete",
            "fingerprint": _historical_fingerprint(stage, inputs),
            "outputs": {str(path): job.pins[path] for path in stage.outputs},
        }:
            raise ReplayError("Original quiet measurement lacks genuine post-command completion")
        verifier.verify("quiet", job.baseline["config"], job.baseline["cuts"])
        expected = _derive(job, data, paths, historical_sources)
        if verify:
            if _read(job.output) != expected:
                raise ReplayError("Retained quiet derivation changed")
        else:
            with job.output.open("xb") as stream:
                stream.write(encode(expected))
        verifier.verify("quiet", job.current, job.output)
        if cache.verify() != {str(path): digest for path, digest in job.pins.items()}:
            raise ReplayError("Retained measurement changed at the terminal boundary")
    return job.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        output = run(args.request.absolute(), verify=args.mode == "verify", sha256=args.sha256)
    except (ReplayError, ValueError, OSError) as error:
        parser.exit(1, f"RETAINED_QUIET_ERROR {error}\n")
    print(output)


if __name__ == "__main__":
    main()
