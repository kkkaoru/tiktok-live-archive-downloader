"""Bind an explicit replacement reflow to the ordinary head/clean producers."""

import argparse
from pathlib import Path

from replay import caption_reflow_verify, clock_contract
from replay.caption_consumer_binding import _path, _read, _ref, _write
from replay.caption_native_continuation import _bindings, _measurement
from replay.caption_token_sources import load_sources
from replay.core import ReplayError
from replay.delivery_captions import read_cuts
from replay.fcp_caption_retarget import face_references
from replay.fcp_transformed_source import clean_source
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record

ROLES = frozenset(
    {
        "voice",
        "voiceReceipt",
        "clockConfig",
        "clockReceipt",
        "reflow",
        "measurementReference",
        "segmentIndex",
    }
)


def prepare(request: Path, output: Path) -> Path:
    """Validate existing artifacts, then write new pinned producer configuration."""
    raw = _read(request)
    if (
        set(raw)
        != {"version", "nativeConfig", "nativePreparationReceipt", "headTemplate", "artifactPaths"}
        or raw.get("version") != 1
    ):
        raise ReplayError("Invalid explicit head continuation schema")
    fixed = _bindings({key: raw[key] for key in ("nativeConfig", "headTemplate")})
    config, template = _read(fixed["nativeConfig"]), _read(fixed["headTemplate"])
    receipt_path = _path(raw["nativePreparationReceipt"])
    prior = _read(receipt_path)
    if prior.get("config") != _ref(fixed["nativeConfig"]):
        raise ReplayError("Native preparation belongs to another config")
    saved = record(prior.get("inputs"))
    if not saved or not all(isinstance(value, str) for value in saved.values()):
        raise ReplayError("Invalid native preparation input pins")
    names = record(raw["artifactPaths"])
    if not ROLES <= set(names) <= ROLES | {"tokenSources"}:
        raise ReplayError("Missing or unknown explicit head artifact role")
    paths = {name: _path(value) for name, value in names.items()}
    tools = _bindings(config.get("inputs"))
    _measurement(tools)
    cuts = _path(record(config.get("futurePaths")).get("cuts"))
    frames = sum(span.frames for span in read_cuts(cuts))
    if prior.get("frames") != frames or prior.get("rate") != 30:
        raise ReplayError("Native preparation and actual cuts differ")
    if saved.get(str(paths["segmentIndex"])) != _ref(paths["segmentIndex"])["sha256"]:
        raise ReplayError("Recognition segment index differs from native preparation")
    pins = {
        str(path): _ref(path)["sha256"]
        for path in (
            request,
            receipt_path,
            *fixed.values(),
            *paths.values(),
            cuts,
            Path(__file__).resolve(),
        )
    }
    snapshot = InputHashSnapshot.capture(
        (Path(name), digest)
        for name, digest in {**saved, **pins}.items()
        if isinstance(digest, str)
    )
    clock = clock_contract.verify(paths["clockConfig"])
    if clock != _read(paths["clockReceipt"]) or clock.get("outputFrames") != frames:
        raise ReplayError("Voice clock receipt or frame count differs")
    clock_refs = record(_read(paths["clockConfig"]).get("artifacts"))
    for role, path in (
        ("outputAudio", paths["voice"]),
        ("receipt", paths["voiceReceipt"]),
        ("cuts", cuts),
        ("audioClock", tools["audioClock"]),
    ):
        if clock_refs.get(role) != _ref(path):
            raise ReplayError("Voice clock uses different selected artifacts")
    caption_reflow_verify.verify(paths["reflow"], source=tools["source"], frames=frames, rate=30)
    old, current = _read(tools["measurementReference"]), _read(paths["measurementReference"])
    if any(old.get(key) != current.get(key) for key in ("style", "expectedPointSize", "width")):
        raise ReplayError("Replacement measurements changed calibrated geometry")
    if (
        record(current.get("inputs")).get(str(tools["nativeExecutable"]))
        != _ref(tools["nativeExecutable"])["sha256"]
    ):
        raise ReplayError("Replacement measurements use another native producer")
    if (
        set(template) != {"version", "faceIndex", "cleanPaths", "name"}
        or template.get("version") != 1
    ):
        raise ReplayError("Invalid head template")
    face = _bindings({"faceIndex": template["faceIndex"]})["faceIndex"]
    references = face_references(face)
    for face_config, _evidence in references:
        observed = _read(face_config)
        if (
            observed.get("source") != str(tools["source"])
            or observed.get("sourceSHA256") != _ref(tools["source"])["sha256"]
        ):
            raise ReplayError("Face observations belong to another source")
    clean_paths = record(template.get("cleanPaths"))
    if set(clean_paths) != {"assemblyManifest", "assemblyReceipt", "cleanIndex"}:
        raise ReplayError("Incomplete cleanup lineage")
    clean: dict[str, object] = {name: _ref(_path(value)) for name, value in clean_paths.items()}
    original, _media, _duration = clean_source(clean)
    if original != tools["source"]:
        raise ReplayError("Cleanup belongs to another source")
    head: dict[str, object] = {
        key: _ref(paths[key]) for key in ("reflow", "measurementReference", "segmentIndex", "voice")
    }
    reflow_sources = record(_read(paths["reflow"]).get("settings")).get("tokenSources")
    if "tokenSources" in paths:
        entries = _read(paths["segmentIndex"]).get("segments")
        if not isinstance(entries, list):
            raise ReplayError("Missing recognized segment entries")
        _sources, source_pins = load_sources(paths["tokenSources"], entries)
        if source_pins != reflow_sources:
            raise ReplayError("Replacement token mapping differs from reflow")
        head["tokenSources"] = _ref(paths["tokenSources"])
    elif reflow_sources is not None:
        raise ReplayError("Reconciled reflow requires its token mapping")
    name = template.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 160:
        raise ReplayError("Invalid explicit project name")
    output = output.absolute()
    if (
        output.exists()
        or output != output.resolve()
        or not output.is_relative_to(Path.cwd() / "edits/work")
    ):
        raise ReplayError("Expected new canonical head work directory")
    head.update(
        {
            "version": 1,
            "faceIndex": _ref(face),
            "cuts": _ref(cuts),
            "nativeExecutable": _ref(tools["nativeExecutable"]),
            "longExecutable": _ref(tools["longExecutable"]),
            "template": _ref(tools["template"]),
            "headReferencePolicy": "automatic-face-relative",
            "collectLayoutFailures": True,
            "terminalFrameCell": True,
            "cleanSource": {**clean, "voiceReceipt": _ref(paths["voiceReceipt"])},
        }
    )
    snapshot.finish()
    output.mkdir(mode=0o700)
    _write(output / "head-config.json", head)
    workflow = _workflow(
        output,
        python=tools["python"],
        name=name,
        inputs=sorted(
            {*saved, *pins, str(face), *[str(_path(value)) for value in clean_paths.values()]}
        ),
    )
    _write(
        output / "binding-receipt.json",
        {
            "version": 1,
            "request": _ref(request),
            "headConfig": _ref(output / "head-config.json"),
            "workflow": _ref(workflow),
            "inputs": pins,
            "nativePreparation": _ref(receipt_path),
            "deliveryComplete": False,
            "observationsRecovered": False,
        },
    )
    return workflow


def _workflow(output: Path, *, python: Path, name: str, inputs: list[str]) -> Path:
    prefix = [str(python), "-B", "-m"]
    config = output / "head-config.json"
    render = output / "render"
    transform, binding = (
        render / "clean-transform-config.json",
        render / "clean-transform-binding.json",
    )
    candidate = output / "clean-semantic.fcpxml"

    def command(phase: str, module: str, args: list[str]) -> list[str]:
        return prefix + [
            "replay.phase_timer",
            str(output / f"{phase}.phases.jsonl"),
            phase,
            "--",
            *prefix,
            module,
            *args,
        ]

    path = output / "workflow.json"
    _write(
        path,
        {
            "version": 1,
            "sourceCaptionProduction": {"bindings": {"clean-semantic": str(binding)}},
            "stages": [
                {
                    "name": "head-semantic",
                    "inputs": [*inputs, str(config)],
                    "outputs": [
                        str(render / "semantic.fcpxml"),
                        str(render / "receipt.json"),
                        str(transform),
                        str(binding),
                    ],
                    "command": command(
                        "head-semantic",
                        "replay.fcp_head_pipeline",
                        [str(config), str(output / "render")],
                    ),
                    "timeout_seconds": 21600,
                },
                {
                    "name": "clean-semantic",
                    "inputs": [
                        *inputs,
                        str(config),
                        str(transform),
                        str(binding),
                        str(render / "semantic.fcpxml"),
                    ],
                    "outputs": [str(candidate), str(output / "clean-semantic.source-lineage.json")],
                    "command": command(
                        "clean-semantic",
                        "replay.fcp_transformed_source",
                        [str(transform), str(candidate), "--name", name],
                    ),
                    "verify_command": prefix
                    + ["replay.fcp_transformed_source", str(transform), str(candidate), "--verify"],
                    "timeout_seconds": 21600,
                },
            ],
        },
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(prepare(args.request, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
