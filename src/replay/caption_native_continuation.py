"""Prepare ordinary native paging from completed cuts and recognition inputs."""

import argparse
import math
from pathlib import Path

from replay import (
    caption_native_inputs,
    caption_reflow_verify,
    caption_style_defaults,
    clock_contract,
)
from replay.caption_consumer_binding import _path, _read, _ref, _write
from replay.caption_font_identity import require_selected_identity, require_selected_name
from replay.caption_native_layout import NativeCaptionStyle
from replay.core import ReplayError
from replay.delivery_captions import read_cuts
from replay.fcp_transformed_source import clean_source
from replay.input_hash_snapshot import InputHashSnapshot
from replay.motion_pre_layout import group_controls
from replay.recordings import record
from replay.stages import file_digest

FIXED_ROLES = frozenset(
    {
        "source",
        "stereoPCM",
        "audioClock",
        "nativeExecutable",
        "longExecutable",
        "measurementReference",
        "fontReceipt",
        "template",
        "python",
        "ffmpeg",
        "codeClosure",
    }
)
MAXIMUM_RATE_CHANGE = 0.5


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReplayError("Expected finite native calibration value")
    return float(value)


def _bindings(raw: object) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, value in record(raw).items():
        ref = record(value)
        raw_path = ref.get("path")
        if name in {"python", "ffmpeg"}:
            if (
                not isinstance(raw_path, str)
                or not Path(raw_path).is_absolute()
                or not Path(raw_path).is_file()
            ):
                raise ReplayError("Expected explicitly pinned executable path")
            path = Path(raw_path)
        else:
            path = _path(raw_path)
        if ref != _ref(path):
            raise ReplayError("Native continuation input pin changed")
        result[name] = path
    return result


def _measurement(paths: dict[str, Path]) -> tuple[list[str], list[Path]]:
    metrics = _read(paths["measurementReference"])
    raw = record(metrics.get("style"))
    name = raw.get("font_name")
    if not isinstance(name, str):
        raise ReplayError("Missing measured native font")
    require_selected_name(name)
    style = NativeCaptionStyle(
        name,
        _number(raw.get("ass_font_size")),
        _number(raw.get("border")),
        _number(raw.get("rim")),
        _number(raw.get("motion_line_advance")),
    )
    point, width = _number(metrics.get("expectedPointSize")), _number(metrics.get("width"))
    if not 1 <= point <= 1024 or not 1 <= width <= 16384:
        raise ReplayError("Invalid measured native dimensions")
    primary = record(_read(paths["fontReceipt"]).get("primary"))
    require_selected_identity(primary)
    if primary.get("pointSize") != point:
        raise ReplayError("Font receipt and measurement point size differ")
    fonts = _read(paths["fontReceipt"]).get("fonts")
    if not isinstance(fonts, list) or not 1 <= len(fonts) <= 64:
        raise ReplayError("Missing measured font asset pins")
    assets = list(_bindings({str(index): value for index, value in enumerate(fonts)}).values())
    recorded = record(metrics.get("inputs"))
    if any(
        recorded.get(str(paths[key])) != file_digest(paths[key])
        for key in ("nativeExecutable", "fontReceipt")
    ):
        raise ReplayError("Calibration does not bind the selected native/font producer")
    if not 0 < paths["template"].stat().st_size <= 8 * 1024 * 1024:
        raise ReplayError("Motion template exceeds bounded metadata size")
    document = paths["template"].read_text(encoding="utf-8")
    group_controls(document)
    return [
        "--ass-font-size",
        str(style.ass_font_size),
        "--point-size",
        str(point),
        "--border",
        str(style.border),
        "--rim",
        str(style.rim),
        "--width",
        str(width),
    ], assets


def prepare(config: Path, output: Path) -> Path:
    """Generate new data and an executable normal graph; do not execute workers."""
    config = _path(str(config.absolute()))
    output = output.absolute()
    if (
        output.exists()
        or output != output.resolve()
        or not output.is_relative_to(Path.cwd() / "edits/work")
    ):
        raise ReplayError("Expected a new canonical native continuation work directory")
    data = _read(config)
    if set(data) != {"version", "inputs", "futurePaths", "evidence"} or data.get("version") != 1:
        raise ReplayError("Invalid native continuation schema")
    paths = _bindings(data.get("inputs"))
    if set(paths) != FIXED_ROLES:
        raise ReplayError("Incomplete fixed native continuation roles")
    future = record(data.get("futurePaths"))
    if set(future) != {"cuts", "nativeConfig"}:
        raise ReplayError("Unexpected future native roles")
    cuts, native_config = _path(future["cuts"]), _path(future["nativeConfig"])
    native = _read(native_config)
    if native.get("cuts") != _ref(cuts) or native.get("source") != _ref(paths["source"]):
        raise ReplayError("Native inputs do not use the selected source/cuts")
    recognition = native_config.parent / "recognition-inputs"
    caption_native_inputs.verify(recognition)
    spans = read_cuts(cuts)
    frames = sum(cut.frames for cut in spans)
    if not 1 <= frames <= 10_000_000 or spans[-1].output_start + spans[-1].frames != frames:
        raise ReplayError("Native cuts must have a complete continuous output clock")
    cut_data = _read(cuts)
    if (
        cut_data.get("fps") != 30
        or record(cut_data.get("audioSupport")).get("recognizedSpeechRemoved") is not False
    ):
        raise ReplayError("Expected audio-supported thirty-fps source cuts")
    if record(cut_data.get("parentSHA256")).get(str(paths["audioClock"])) != file_digest(
        paths["audioClock"]
    ):
        raise ReplayError("Native cuts and tempo source clock differ")
    options, assets = _measurement(paths)
    evidence = list(_bindings(data.get("evidence")).values())
    if not evidence or paths["fontReceipt"] not in evidence:
        raise ReplayError("Native evidence must include the selected font receipt")
    source_files = [Path(__file__).resolve(), Path(caption_style_defaults.__file__).resolve()]
    code = record(_read(paths["codeClosure"]).get("inputs"))
    code_paths = _bindings(
        {name: {"path": name, "sha256": digest} for name, digest in code.items()}
    )
    if str(Path(caption_native_inputs.__file__).resolve()) not in code_paths:
        raise ReplayError("Native continuation code closure is incomplete")
    tool_targets = {name: paths[name].resolve(strict=True) for name in ("python", "ffmpeg")}
    bound = {
        str(path.resolve(strict=True)): file_digest(path)
        for path in (
            config,
            cuts,
            native_config,
            recognition / "segments.json",
            recognition / "receipt.json",
            *paths.values(),
            *assets,
            *evidence,
            *source_files,
            *code_paths.values(),
        )
    }
    snapshot = InputHashSnapshot.capture((Path(name), sha) for name, sha in bound.items())
    output.mkdir(mode=0o700)
    selection = output / "tempo-selection.json"
    _write(
        selection,
        {
            "version": 1,
            "coordinateSpace": "native-presentation-seconds",
            "source": _ref(paths["stereoPCM"]),
            "clock": _ref(paths["audioClock"]),
            "intervals": [
                {
                    "start": {"value": cut.source_start, "timescale": 30},
                    "end": {"value": cut.source_end, "timescale": 30},
                }
                for cut in spans
            ],
            "evidence": [_ref(cuts)],
        },
    )
    python = [str(paths["python"]), "-B", "-m"]
    stages = _tempo_stages(
        python=python,
        bound=bound,
        selection=selection,
        output=output,
        config=config,
        ffmpeg=paths["ffmpeg"],
    )
    paging = output / "paging"
    command = (
        python
        + [
            "replay.caption_native_paging",
            str(recognition),
            str(paging),
            str(paths["nativeExecutable"]),
            "--source-media",
            str(paths["source"]),
            "--long-executable",
            str(paths["longExecutable"]),
            "--collect-layout-failures",
            # The recording pipeline's Motion captions are never scaled: page instead.
            "--split-wide",
        ]
        + options
    )
    for item in evidence:
        command.extend(("--evidence", str(item)))
    workflow = output / "workflow.json"
    _write(
        workflow,
        {
            "version": 1,
            "stages": stages
            + [
                {
                    "name": "native-pos-paging",
                    "inputs": list(bound),
                    "outputs": [str(paging / "reflow.json")],
                    "command": python
                    + [
                        "replay.phase_timer",
                        str(output / "paging.phases.jsonl"),
                        "native-pos-paging",
                        "--",
                    ]
                    + command,
                    "verify_command": python
                    + [
                        "replay.caption_reflow_verify",
                        str(paging / "reflow.json"),
                        "--source-media",
                        str(paths["source"]),
                        "--frames",
                        str(frames),
                    ],
                    "timeout_seconds": 21600,
                }
            ],
        },
    )
    snapshot.finish()
    if any(paths[name].resolve(strict=True) != target for name, target in tool_targets.items()):
        raise ReplayError("Explicit executable target changed during preparation")
    _write(
        output / "receipt.json",
        {
            "version": 1,
            "config": _ref(config),
            "workflow": _ref(workflow),
            "inputs": bound,
            "frames": frames,
            "rate": 30,
            "tempoSelection": _ref(selection),
            "style": caption_style_defaults.DEFAULT_CAPTION_STYLE_ID,
            "measurementsPerformed": False,
            "deliveryComplete": False,
        },
    )
    return workflow


def _tempo_stages(
    *,
    python: list[str],
    bound: dict[str, str],
    selection: Path,
    output: Path,
    config: Path,
    ffmpeg: Path,
) -> list[dict[str, object]]:
    return [
        {
            "name": "cut-tempo-audio",
            "inputs": [*bound, str(selection)],
            "outputs": [str(output / "tempo/candidate.wav"), str(output / "tempo/receipt.json")],
            "command": python
            + [
                "replay.phase_timer",
                str(output / "tempo.phases.jsonl"),
                "cut-tempo-audio",
                "--",
                *python,
                "replay.clocked_tempo_audio",
                str(selection),
                str(output / "tempo"),
                "--sha256",
                file_digest(selection),
                "--ffmpeg",
                str(ffmpeg),
                "--maximum-rate-change",
                str(MAXIMUM_RATE_CHANGE),
            ],
            "timeout_seconds": 21600,
        },
        {
            "name": "tempo-clock-contract",
            "inputs": [
                *bound,
                str(selection),
                str(output / "tempo/candidate.wav"),
                str(output / "tempo/receipt.json"),
            ],
            "outputs": [str(output / "tempo-clock-config.json")],
            "command": python
            + [
                "replay.phase_timer",
                str(output / "tempo-clock.phases.jsonl"),
                "tempo-clock-contract",
                "--",
                *python,
                "replay.caption_native_continuation",
                str(config),
                str(output),
                "--tempo-contract",
            ],
            "verify_command": python
            + ["replay.clock_contract", str(output / "tempo-clock-config.json"), "--inspect"],
            "timeout_seconds": 1800,
        },
        {
            "name": "tempo-complete-pcm-proof",
            "inputs": [
                *bound,
                str(output / "tempo-clock-config.json"),
                str(output / "tempo/candidate.wav"),
                str(output / "tempo/receipt.json"),
            ],
            "outputs": [str(output / "tempo-clock-receipt.json")],
            "command": python
            + [
                "replay.phase_timer",
                str(output / "tempo-proof.phases.jsonl"),
                "tempo-complete-pcm-proof",
                "--",
                *python,
                "replay.clock_contract",
                str(output / "tempo-clock-config.json"),
                "--output",
                str(output / "tempo-clock-receipt.json"),
            ],
            "verify_command": [
                str(ffmpeg),
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(output / "tempo/candidate.wav"),
                "-f",
                "null",
                "-",
            ],
            "timeout_seconds": 1800,
        },
    ]


def prepare_tempo_contract(config: Path, output: Path) -> Path:
    """Bind actual tempo output into the existing one-application verifier."""
    paths = _bindings(_read(config).get("inputs"))
    future = record(_read(config).get("futurePaths"))
    artifacts = {
        "producerConfig": output / "tempo-selection.json",
        "producerReceipt": output / "tempo/receipt.json",
        "inputAudio": paths["stereoPCM"],
        "outputAudio": output / "tempo/candidate.wav",
        "audioClock": paths["audioClock"],
        "cuts": _path(future.get("cuts")),
    }
    destination = output / "tempo-clock-config.json"
    _write(
        destination,
        {
            "version": 1,
            "operation": "tempo-cut-audio",
            "previousApplications": 0,
            "inputClock": "continuous-decoded-pcm",
            "outputClock": "edited-cut-pcm",
            "artifacts": {name: _ref(path) for name, path in artifacts.items()},
        },
    )
    clock_contract.verify(destination)
    return destination


def prepare_head(config: Path, output: Path, *, template: Path) -> Path:
    """Bind completed paging/voice/cleanup into the existing head and clean graph."""
    inputs = _bindings(_read(config).get("inputs"))
    future = record(_read(config).get("futurePaths"))
    prepared = _read(output / "receipt.json")
    if prepared.get("config") != _ref(config):
        raise ReplayError("Native preparation config changed before head binding")
    saved_inputs = record(prepared.get("inputs"))
    if not saved_inputs or not all(isinstance(value, str) for value in saved_inputs.values()):
        raise ReplayError("Invalid native preparation input pins")
    snapshot = InputHashSnapshot.capture(
        (Path(name), digest) for name, digest in saved_inputs.items() if isinstance(digest, str)
    )
    saved = _read(template)
    if set(saved) != {"version", "faceIndex", "cleanPaths", "name"} or saved.get("version") != 1:
        raise ReplayError("Invalid head continuation template")
    face = _bindings({"faceIndex": saved.get("faceIndex")})["faceIndex"]
    clean_paths = record(saved.get("cleanPaths"))
    if set(clean_paths) != {"assemblyManifest", "assemblyReceipt", "cleanIndex"}:
        raise ReplayError("Incomplete completed cleanup roles")
    clean: dict[str, object] = {name: _ref(_path(value)) for name, value in clean_paths.items()}
    original, clean_media, _duration = clean_source(clean)
    if original != inputs["source"]:
        raise ReplayError("Completed cleanup belongs to a different source")
    clock = clock_contract.verify(output / "tempo-clock-config.json")
    frames = prepared.get("frames")
    if type(frames) is not int or clock.get("outputFrames") != frames:
        raise ReplayError("Completed tempo and caption frame counts differ")
    reflow = output / "paging/reflow.json"
    caption_reflow_verify.verify(reflow, source=original, frames=frames, rate=30)
    name = saved.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 160:
        raise ReplayError("Expected bounded explicit project name")
    cuts, native = _path(future.get("cuts")), _path(future.get("nativeConfig"))
    head = {
        "version": 1,
        "faceIndex": _ref(face),
        "cuts": _ref(cuts),
        "reflow": _ref(reflow),
        "segmentIndex": _ref(native.parent / "recognition-inputs/segments.json"),
        "measurementReference": _ref(output / "paging/metrics-0000/inputs.json"),
        "nativeExecutable": _ref(inputs["nativeExecutable"]),
        "longExecutable": _ref(inputs["longExecutable"]),
        "template": _ref(inputs["template"]),
        "voice": _ref(output / "tempo/candidate.wav"),
        "headReferencePolicy": "automatic-face-relative",
        "collectLayoutFailures": True,
        "terminalFrameCell": True,
        "cleanSource": {**clean, "voiceReceipt": _ref(output / "tempo/receipt.json")},
    }
    head_config = output / "head-config.json"
    _write(head_config, head)
    run = output / "head"
    python = [str(inputs["python"]), "-B", "-m"]
    transform = run / "clean-transform-config.json"
    binding = run / "clean-transform-binding.json"
    candidate = run / "clean-semantic.fcpxml"
    common = [
        config,
        template,
        head_config,
        face,
        cuts,
        reflow,
        output / "receipt.json",
        output / "tempo-clock-config.json",
        output / "tempo/candidate.wav",
        output / "tempo/receipt.json",
        inputs["template"],
        clean_media,
    ]
    common.extend(_path(value) for value in clean_paths.values())
    common.extend(Path(path) for path in record(prepared.get("inputs")))
    workflow = output / "head-workflow.json"
    _write(
        workflow,
        {
            "version": 1,
            "sourceCaptionProduction": {"bindings": {"clean-semantic": str(binding)}},
            "stages": [
                {
                    "name": "head-semantic",
                    "inputs": sorted({str(path) for path in common}),
                    "outputs": [
                        str(run / "semantic.fcpxml"),
                        str(run / "receipt.json"),
                        str(transform),
                        str(binding),
                    ],
                    "command": python
                    + [
                        "replay.phase_timer",
                        str(output / "head.phases.jsonl"),
                        "head-semantic",
                        "--",
                        *python,
                        "replay.fcp_head_pipeline",
                        str(head_config),
                        str(run),
                    ],
                    "timeout_seconds": 21600,
                },
                {
                    "name": "clean-semantic",
                    "inputs": [
                        str(transform),
                        str(binding),
                        str(run / "semantic.fcpxml"),
                        *sorted({str(path) for path in common}),
                    ],
                    "outputs": [str(candidate), str(run / "clean-semantic.source-lineage.json")],
                    "command": python
                    + [
                        "replay.phase_timer",
                        str(output / "clean.phases.jsonl"),
                        "clean-semantic",
                        "--",
                        *python,
                        "replay.fcp_transformed_source",
                        str(transform),
                        str(candidate),
                        "--name",
                        name,
                    ],
                    "verify_command": python
                    + ["replay.fcp_transformed_source", str(transform), str(candidate), "--verify"],
                    "timeout_seconds": 21600,
                },
            ],
        },
    )
    snapshot.finish()
    return workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tempo-contract", action="store_true")
    mode.add_argument("--head-template", type=Path)
    args = parser.parse_args(argv)
    if args.head_template is not None:
        print(prepare_head(args.config, args.output, template=args.head_template))
    else:
        operation = prepare_tempo_contract if args.tempo_contract else prepare
        print(operation(args.config, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
