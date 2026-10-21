"""Plan the first three prerequisites of any initialized replay, without executing them.

Prepare requires a new work directory. Resume the resulting immutable workflow
with replay.workflow and its original state directory; partial artifacts are not
adopted by this factory. Clock checks and separated PCM checks are technical
evidence, never perceptual or finished-delivery approval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from replay.audio_clock import AudioFrameClock
from replay.bgm_separation import MODEL_CONFIG_SHA256, MODEL_NAME, MODEL_REVISION, MODEL_SHA256
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.delivery_audio import separate, verify_material
from replay.delivery_media import MAX_CLOCK_BYTES, parse_clock
from replay.delivery_media import main as media_main
from replay.edit_config import CONFIG_NAME, load_config
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.stages import StageError, file_digest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "replay-preparation-v1"
MAX_METADATA_BYTES = 1024 * 1024
MAX_INPUTS = 220
TOOL_NAMES = frozenset({"python", "ffmpeg", "ffprobe", "env"})
# A derived source (re-encoded for native decoding, given a lead-in) records its origin.
DERIVATION_RECORDS = ("decodable.json", "lead-in.json")
SETTINGS_FIELDS = frozenset(
    {"version", "tools", "hfHome", "device", "clockTimeoutSeconds", "separationTimeoutSeconds"}
)
PLAN_FIELDS = frozenset(
    {
        "version",
        "protocol",
        "editRoot",
        "settings",
        "workRoot",
        "inputs",
        "candidateOnly",
        "deliveryComplete",
    }
)
Kind = Literal["video", "audio", "separation"]


@dataclass(frozen=True, slots=True)
class Pin:
    path: Path
    sha256: str

    def data(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class Settings:
    tools: dict[str, Tool]
    hf_home: Path
    device: Literal["cpu", "mps"]
    clock_timeout: int
    separation_timeout: int
    runtime_inputs: tuple[Pin, ...] = ()


@dataclass(frozen=True, slots=True)
class Preparation:
    edit_root: Path
    settings_path: Path
    work: Path
    settings: Settings
    source: Path
    pins: tuple[Pin, ...]


@dataclass(frozen=True, slots=True)
class Tool:
    path: Path
    pin: Pin


def _tool(value: object) -> Tool:
    data = record(value)
    _require(set(data) == {"path", "sha256"}, "Invalid tool pin fields")
    raw = data["path"]
    _require(isinstance(raw, str) and bool(raw), "Missing tool invocation path")
    path = Path(str(raw))
    _require(
        path.is_absolute() and str(path) == raw and ".." not in path.parts,
        "Invalid tool invocation path",
    )
    resolved = path.resolve(strict=True)
    return Tool(path, _pin({"path": str(resolved), "sha256": data["sha256"]}))


def _python_config(tool: Tool) -> Path:
    # Preserve the bin/python invocation alias so Python selects this venv.
    _local(tool.path, ROOT)
    config = _path(str(tool.path.parent.parent / "pyvenv.cfg"))
    _require(
        tool.path.parent.name == "bin" and config.is_file(), "Use a pinned repository Python venv"
    )
    version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    _require(
        re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is not None,
        "Expected configured Python patch version",
    )
    info = config.stat()
    _require(0 < info.st_size <= 65536, "Invalid venv configuration size")
    values = dict(
        line.strip().split(" = ", 1)
        for line in config.read_text(encoding="utf-8").splitlines()
        if " = " in line
    )
    _require(
        values.get("version_info", values.get("version")) == version,
        "Python venv patch version differs from the configured version",
    )
    return config


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _path(value: object) -> Path:
    _require(isinstance(value, str) and bool(value), "Expected absolute path")
    path = Path(str(value))
    _require(
        path.is_absolute() and str(path) == value and path == path.resolve(),
        "Expected canonical unlinked path",
    )
    return path


def _local(path: Path, parent: Path) -> None:
    _require(path.is_relative_to(parent) and path != parent, "Path escapes repository scope")


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        _require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _read(path: Path, *, budget: int = MAX_METADATA_BYTES) -> object:
    _path(str(path))
    info = path.lstat()
    _require(
        stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and 0 < info.st_size <= budget,
        "Expected bounded regular single-link metadata",
    )
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        _require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (info.st_dev, info.st_ino, info.st_size),
            "Metadata changed before opening",
        )
        data = stream.read(budget + 1)
    _require(len(data) == info.st_size, "Metadata changed or exceeds its budget")
    after = path.lstat()
    _require(
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "Metadata changed during reading",
    )
    return json.loads(data, object_pairs_hook=_pairs)


def _pin(value: object) -> Pin:
    data = record(value)
    _require(set(data) == {"path", "sha256"}, "Invalid input pin fields")
    digest = data["sha256"]
    _require(
        isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest) is not None,
        "Invalid SHA256 pin",
    )
    return Pin(_path(data["path"]), str(digest))


def _timeout(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 86400:
        raise ReplayError("Timeout must be integral and within one day")
    return value


def read_settings(path: Path) -> Settings:
    """Read explicit tools and an already-installed, repository-local model cache."""
    _local(_path(str(path)), ROOT / "edits" / "work")
    data = record(_read(path))
    _require(
        set(data) - {"runtimeInputs"} == SETTINGS_FIELDS
        and type(data["version"]) is int
        and data["version"] == 1,
        "Invalid preparation settings",
    )
    tools = record(data["tools"])
    _require(
        TOOL_NAMES <= set(tools) <= TOOL_NAMES | {"separationPython"},
        "Missing or unknown preparation tool",
    )
    checked = {name: _tool(value) for name, value in tools.items()}
    _require(
        all(os.access(item.path, os.X_OK) for item in checked.values()),
        "Preparation tools must be executable",
    )
    _python_config(checked["python"])
    if "separationPython" in checked:
        _python_config(checked["separationPython"])
    home = _path(data["hfHome"])
    _local(home, ROOT / "edits" / "models")
    _require(home.is_dir(), "Model cache must already exist; no download is permitted")
    device = data["device"]
    if device not in ("cpu", "mps"):
        raise ReplayError("Choose an explicit supported separation device")
    runtime = data.get("runtimeInputs", [])
    if not isinstance(runtime, list) or len(runtime) > 32:
        raise ReplayError("Expected bounded explicit runtime input pins")
    runtime_pins = tuple(_pin(value) for value in runtime)
    _require(
        len({item.path for item in runtime_pins}) == len(runtime_pins), "Duplicate runtime input"
    )
    for item in runtime_pins:
        _local(item.path, ROOT)
    return Settings(
        checked,
        home,
        "cpu" if device == "cpu" else "mps",
        _timeout(data["clockTimeoutSeconds"]),
        _timeout(data["separationTimeoutSeconds"]),
        runtime_pins,
    )


def _models(settings: Settings) -> tuple[Pin, Pin]:
    folder = (
        settings.hf_home
        / "hub"
        / ("models--" + MODEL_NAME.replace("/", "--"))
        / "snapshots"
        / MODEL_REVISION
    )
    weights = (folder / "model.safetensors").resolve(strict=True)
    config = (folder / "config.json").resolve(strict=True)
    _local(weights, settings.hf_home)
    _local(config, settings.hf_home)
    # The backend uses the lexical snapshot parent, including HF blob aliases.
    return Pin(weights, MODEL_SHA256), Pin(config, MODEL_CONFIG_SHA256)


def _code_paths() -> tuple[Path, ...]:
    """Inventory transitive local imports, including lazily imported CPU helpers."""
    return tuple(
        item.path
        for item in capture(
            (
                "replay_preparation_plan",
                "delivery_media",
                "delivery_audio",
                "phase_timer",
            ),
            package=ROOT / "src" / "replay",
        )
    )


def _derived_source(original: Path, source: Path) -> Path:
    """``source`` when its derivation records lead back to the edit's original, unchanged."""
    current = source
    for _ in range(len(DERIVATION_RECORDS) + 1):
        if current == original:
            return source
        records = [current.parent / name for name in DERIVATION_RECORDS]
        found = [path for path in records if path.is_file()]
        _require(len(found) == 1, f"{current} has no single derivation record")
        data = record(_read(found[0]))
        _require(
            data.get("output") == str(current)
            and data.get("outputSHA256") == file_digest(current)
            and isinstance(data.get("source"), str),
            f"{found[0]} does not record {current} unchanged",
        )
        current = Path(str(data["source"]))
    raise ReplayError("The source is not a recorded derivation of the edit's original")


def _expected(
    edit_root: Path, settings_path: Path, settings: Settings, source: Path | None = None
) -> tuple[Path, tuple[Pin, ...]]:
    _local(_path(str(edit_root)), ROOT / "edits")
    config = load_config(edit_root)
    media = config.source if source is None else _derived_source(config.source, source)
    references = (
        *(tool.pin for tool in settings.tools.values()),
        *_models(settings),
        *settings.runtime_inputs,
    )
    supplied: dict[Path, str] = {}
    for item in references:
        _require(
            item.path not in supplied or supplied[item.path] == item.sha256,
            "Conflicting tool/model role pins",
        )
        supplied[item.path] = item.sha256
    paths = (
        edit_root / CONFIG_NAME,
        settings_path,
        media,
        *_code_paths(),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / ".python-version",
        _python_config(settings.tools["python"]),
        _python_config(settings.tools.get("separationPython", settings.tools["python"])),
    )
    for path in paths:
        supplied.setdefault(path, file_digest(path))
    _require(len(supplied) <= MAX_INPUTS, "Preparation input closure exceeds workflow budget")
    return media, tuple(Pin(path, digest) for path, digest in sorted(supplied.items()))


def _environment(preparation: Preparation) -> tuple[str, ...]:
    work, home = preparation.work, preparation.settings.hf_home
    return (
        str(preparation.settings.tools["env"].path),
        f"TMPDIR={work / 'scratch'}",
        f"TMP={work / 'scratch'}",
        f"TEMP={work / 'scratch'}",
        f"XDG_CACHE_HOME={work / 'cache'}",
        f"HF_HOME={home}",
        f"HF_HUB_CACHE={home / 'hub'}",
        "HF_HUB_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={ROOT / 'src'}",
    )


def _python(
    preparation: Preparation, module: str, *arguments: str, separation: bool = False
) -> tuple[str, ...]:
    return (
        str(
            preparation.settings.tools.get(
                "separationPython", preparation.settings.tools["python"]
            ).path
            if separation
            else preparation.settings.tools["python"].path
        ),
        "-B",
        "-m",
        f"replay.{module}",
        *arguments,
    )


def _workflow(preparation: Preparation) -> dict[str, object]:
    work, settings = preparation.work, preparation.settings
    manifest = work / "preparation.json"
    clocks = (work / "video-clock.json", work / "audio-clock.json")
    common = [str(item.path) for item in preparation.pins] + [str(manifest)]
    stages: list[dict[str, object]] = []
    for kind, output in zip(("video", "audio"), clocks, strict=True):
        command = _python(preparation, "replay_preparation_plan", "produce", str(manifest), kind)
        stages.append(
            {
                "name": f"source-{kind}-clock",
                "command": [
                    *_environment(preparation),
                    *_python(
                        preparation,
                        "phase_timer",
                        str(work / "phases.jsonl"),
                        f"source-{kind}-clock",
                        "--",
                        *command,
                    ),
                ],
                "inputs": common,
                "outputs": [str(output), str(work / f"{kind}-clock.producer.json")],
                "timeout_seconds": settings.clock_timeout,
                "verify_command": list(
                    _python(
                        preparation,
                        "replay_preparation_plan",
                        "verify-media-clocks",
                        str(manifest),
                        kind,
                    )
                ),
            }
        )
    separation = work / "separation"
    outputs = [str(path) for path in _outputs(preparation, "separation")]
    outputs.append(str(separation / "producer.json"))
    command = _python(
        preparation,
        "replay_preparation_plan",
        "produce",
        str(manifest),
        "separation",
        separation=True,
    )
    stages.append(
        {
            "name": "source-voice-separation",
            "command": [
                *_environment(preparation),
                *_python(
                    preparation,
                    "phase_timer",
                    str(work / "phases.jsonl"),
                    "source-voice-separation",
                    "--",
                    *command,
                ),
            ],
            "inputs": [
                *common,
                *map(str, clocks),
                str(work / "video-clock.producer.json"),
                str(work / "audio-clock.producer.json"),
            ],
            "outputs": outputs,
            "timeout_seconds": settings.separation_timeout,
            "verify_command": list(
                _python(
                    preparation,
                    "replay_preparation_plan",
                    "verify-media-clocks",
                    str(manifest),
                    "separation",
                )
            ),
        }
    )
    workflow: dict[str, object] = {"version": 1, "stages": stages}
    return workflow


def _write(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(
    edit_root: Path, settings_path: Path, work: Path, *, source: Path | None = None
) -> Path:
    """Create a non-overwriting plan only; replay.workflow owns execution and resume.

    ``source`` names a recorded derivation of the edit's original (a scaled or lead-in copy) to
    prepare instead of the original itself.
    """
    _local(_path(str(work)), ROOT / "edits" / "work")
    _require(
        work.parent.is_dir() and not work.exists(),
        "Use a new work root with an existing parent; inspect occupied/partial results",
    )
    settings = read_settings(settings_path)
    media, pins = _expected(edit_root, settings_path, settings, source)
    snapshot = InputHashSnapshot.capture((item.path, item.sha256) for item in pins)
    preparation = Preparation(edit_root, settings_path, work, settings, media, pins)
    workflow = _workflow(preparation)
    snapshot.finish()
    _resolutions(preparation)
    work.mkdir(mode=0o700)
    (work / "scratch").mkdir(mode=0o700)
    (work / "cache").mkdir(mode=0o700)
    _write(
        work / "preparation.json",
        {
            "version": 1,
            "protocol": PROTOCOL,
            "editRoot": str(edit_root),
            "settings": str(settings_path),
            "workRoot": str(work),
            "inputs": [item.data() for item in pins],
            "candidateOnly": True,
            "deliveryComplete": False,
            **({} if source is None else {"source": str(media)}),
        },
    )
    _write(work / "workflow.json", workflow)
    return work / "workflow.json"


def _load(path: Path) -> Preparation:
    data = record(_read(path))
    _require(
        set(data) - {"source"} == PLAN_FIELDS
        and type(data["version"]) is int
        and data["version"] == 1
        and data["protocol"] == PROTOCOL
        and data["candidateOnly"] is True
        and data["deliveryComplete"] is False,
        "Invalid preparation manifest",
    )
    edit_root, settings_path, work = (
        _path(data["editRoot"]),
        _path(data["settings"]),
        _path(data["workRoot"]),
    )
    _local(work, ROOT / "edits" / "work")
    _require(
        path == work / "preparation.json", "Preparation manifest is not in its original work root"
    )
    raw = data["inputs"]
    _require(
        isinstance(raw, list) and 1 <= len(raw) <= MAX_INPUTS, "Expected bounded preparation pins"
    )
    if not isinstance(raw, list):
        raise ReplayError("Invalid preparation pins")
    pins = tuple(_pin(item) for item in raw)
    _require(len({item.path for item in pins}) == len(pins), "Duplicate preparation input")
    settings = read_settings(settings_path)
    chosen = _path(data["source"]) if "source" in data else None
    source, expected = _expected(edit_root, settings_path, settings, chosen)
    _require(pins == expected, "Preparation input hashes or dependency membership changed")
    _require(
        record(_read(work / "workflow.json"))
        == _workflow(Preparation(edit_root, settings_path, work, settings, source, pins)),
        "Workflow no longer matches the pinned preparation",
    )
    return Preparation(edit_root, settings_path, work, settings, source, pins)


def _resolutions(preparation: Preparation) -> None:
    _require(
        read_settings(preparation.settings_path) == preparation.settings,
        "Tool/cache alias or runtime settings changed",
    )
    _require(
        all(item in preparation.pins for item in _models(preparation.settings)),
        "Model cache alias changed",
    )
    original = load_config(preparation.edit_root).source
    _require(
        _derived_source(original, preparation.source) == preparation.source,
        "Source alias changed",
    )


def _clock(path: Path, kind: Literal["video", "audio"]) -> None:
    data = _read(path, budget=MAX_CLOCK_BYTES)
    if kind == "audio":
        AudioFrameClock.from_probe(data)
    else:
        clock = parse_clock(data)
        frames = record(data).get("frames")
        _require(
            isinstance(frames, list)
            and all(
                type(record(frame).get("duration")) is int
                and int(str(record(frame)["duration"])) > 0
                for frame in frames
            ),
            "Invalid decoded video frame duration",
        )
        _require(clock.time_base <= 1, "Invalid decoded video time base")


def _outputs(preparation: Preparation, kind: Kind) -> tuple[Path, ...]:
    if kind in ("video", "audio"):
        return (preparation.work / f"{kind}-clock.json",)
    names = ["model-input.wav", "vocals.wav", "result.json"]
    if preparation.settings.device == "cpu":
        names.extend(
            f"{phase}-descriptors.json"
            for phase in ("before-model-import", "after-model-import", "after-separation")
        )
    return tuple(preparation.work / "separation" / name for name in names)


def _producer_path(preparation: Preparation, kind: Kind) -> Path:
    return (
        preparation.work / "separation" / "producer.json"
        if kind == "separation"
        else preparation.work / f"{kind}-clock.producer.json"
    )


def _producer_data(preparation: Preparation, kind: Kind) -> dict[str, object]:
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "kind": kind,
        "preparationSHA256": file_digest(preparation.work / "preparation.json"),
        "source": next(item.data() for item in preparation.pins if item.path == preparation.source),
        "outputs": {str(path): file_digest(path) for path in _outputs(preparation, kind)},
        "sharedProducerCompleted": True,
        "candidateOnly": True,
        "deliveryComplete": False,
    }


def _verify_producer(preparation: Preparation, kind: Kind) -> None:
    _require(
        record(_read(_producer_path(preparation, kind))) == _producer_data(preparation, kind),
        "Missing or changed completed shared producer evidence",
    )


def _domain(preparation: Preparation, kind: Kind) -> None:
    if kind in ("video", "audio"):
        _clock(preparation.work / f"{kind}-clock.json", kind)
    else:
        _clock(preparation.work / "video-clock.json", "video")
        audio = AudioFrameClock.from_probe(
            _read(preparation.work / "audio-clock.json", budget=MAX_CLOCK_BYTES)
        )
        directory = preparation.work / "separation"
        receipt = record(_read(directory / "result.json"))
        _require(
            receipt.get("device") == preparation.settings.device
            and receipt.get("preparedInput") is False
            and receipt.get("loudnessNormalization") is False,
            "Separation policy changed",
        )
        verify_material(preparation.source, directory)
        with wave.open(str(directory / "model-input.wav"), "rb") as stream:
            _require(
                abs(stream.getnframes() * audio.sample_rate - audio.sample_ends[-1] * 44100)
                <= audio.sample_rate,
                "Separated input is not on the decoded source sample clock",
            )


def produce(path: Path, kind: Kind) -> None:
    """Call shared producers once and seal evidence only after successful completion."""
    preparation = _load(path)
    outputs = (*_outputs(preparation, kind), _producer_path(preparation, kind))
    _require(
        not any(output.exists() or output.is_symlink() for output in outputs),
        "Producer outputs are occupied; reconcile the retained attempt",
    )
    snapshot = InputHashSnapshot.capture((item.path, item.sha256) for item in preparation.pins)
    if kind == "separation":
        _verify_producer(preparation, "video")
        _verify_producer(preparation, "audio")
        separate(
            preparation.source,
            preparation.work / "separation",
            ffmpeg=str(preparation.settings.tools["ffmpeg"].path),
            device=preparation.settings.device,
        )
    else:
        code = media_main(
            [
                "clock" if kind == "video" else "audio-clock",
                str(preparation.source),
                str(preparation.work / f"{kind}-clock.json"),
                "--ffprobe",
                str(preparation.settings.tools["ffprobe"].path),
            ]
        )
        _require(code == 0, "Shared clock producer failed; partial output is retained")
    _domain(preparation, kind)
    snapshot.finish()
    _resolutions(preparation)
    _write(_producer_path(preparation, kind), _producer_data(preparation, kind))


def verify_media_clocks(path: Path, kind: Kind) -> None:
    """Read-only source-bound domain checks, with true-content endpoint pin checks."""
    preparation = _load(path)
    snapshot = InputHashSnapshot.capture((item.path, item.sha256) for item in preparation.pins)
    _verify_producer(preparation, kind)
    if kind == "separation":
        _verify_producer(preparation, "video")
        _verify_producer(preparation, "audio")
    _domain(preparation, kind)
    snapshot.finish()
    _resolutions(preparation)


class Arguments(argparse.Namespace):
    action: str
    edit_root: Path
    settings: Path
    work: Path
    plan: Path
    kind: Kind
    source: Path | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("prepare")
    plan.add_argument("edit_root", type=Path)
    plan.add_argument("settings", type=Path)
    plan.add_argument("work", type=Path)
    plan.add_argument("--source", type=Path, help="a recorded derivation of the original")
    verify = commands.add_parser("verify-media-clocks")
    verify.add_argument("plan", type=Path)
    verify.add_argument("kind", choices=("video", "audio", "separation"))
    producer = commands.add_parser("produce")
    producer.add_argument("plan", type=Path)
    producer.add_argument("kind", choices=("video", "audio", "separation"))
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    try:
        if args.action == "prepare":
            print(prepare(args.edit_root, args.settings, args.work, source=args.source))
        elif args.action == "produce":
            produce(args.plan, args.kind)
        else:
            verify_media_clocks(args.plan, args.kind)
            print(f"Verified {args.kind}; delivery remains unapproved")
    except (ReplayError, StageError, OSError, ValueError, SyntaxError, wave.Error) as error:
        print(f"REPLAY_PREPARATION_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
