"""Issue provisional v1 envelopes from current static geometry and fresh native metrics.

No colour/outline/gradient approval is required or granted. The font asset is a
pinned historical asset, not proof of the file actually selected by CoreText.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

from replay import motion_calibration_verify as audit
from replay import motion_native_envelope as envelope
from replay.caption_native_layout import NativeCaptionLayout
from replay.recordings import record
from replay.stages import execute_command


def _reference(inputs: audit.Inputs, value: object) -> Path:
    raw = record(value)
    if set(raw) != {"path", "sha256"}:
        raise ValueError("Expected a pinned geometry candidate reference")
    path = inputs.path(raw["path"])
    if inputs.bind(path) != raw["sha256"]:
        raise ValueError("Geometry candidate reference changed")
    return path


def prepare(config: Path, directory: Path) -> Path:
    """Remeasure only the calibration strings; never approve unseen glyphs or delivery.

    The v1 envelope is already provisional. Keep its seven-key contract unchanged;
    attach the narrower scope and unresolved historical provenance as evidence.
    Existing outputs and failed attempts are never overwritten.
    """
    inputs = audit.Inputs()
    raw = inputs.json(config)
    if set(raw) != {"geometryConfig", "geometryReceipt", "native", "fontAsset"}:
        raise ValueError("Invalid geometry candidate configuration")
    paths = {key: _reference(inputs, value) for key, value in raw.items()}
    previous = inputs.json(paths["geometryReceipt"])
    if previous.get("scope") != "geometry-only" or previous.get("geometryContained") is not True:
        raise ValueError("Passing geometry-only evidence is required")
    for name, digest in record(previous.get("inputs")).items():
        _reference(inputs, {"path": name, "sha256": digest})
    if str(paths["geometryConfig"]) not in record(previous["inputs"]):
        raise ValueError("Geometry receipt does not bind the configuration")
    geometry_config = inputs.json(paths["geometryConfig"])
    geometry_paths = record(geometry_config.get("paths"))
    historical = inputs.json(inputs.path(geometry_paths.get("calibration")))
    references = historical.get("evidence")
    asset_pin = {"path": str(paths["fontAsset"]), "sha256": inputs.bind(paths["fontAsset"])}
    if not isinstance(references, list) or asset_pin not in references:
        raise ValueError("Font asset is not bound by historical calibration")
    style = envelope._style(historical.get("style"))
    font = envelope._font(historical.get("nativeFont"))
    canvas = envelope._canvas(historical.get("canvas"))
    padded_request = inputs.json(inputs.path(geometry_paths.get("paddedRequest")))
    padded_response = inputs.json(inputs.path(geometry_paths.get("paddedResponse")))
    sizes = audit.native_sizes(padded_request, padded_response)
    plan = audit.parse_plan(inputs.json(inputs.path(geometry_paths.get("plan"))))
    texts = tuple(dict.fromkeys(line for state in plan.states for line in state.text.split("\n")))
    if not 1 <= len(texts) <= envelope.MAX_PLAN_TEXTS:
        raise ValueError("Too many calibration strings")
    inputs.bind(Path(__file__))
    native_sources = Path.cwd() / "native" / "apple-pro-apps" / "Sources"
    for name in (
        "ProAppsCore/CaptionTextMeasurement.swift",
        "ProAppsCore/StyledCaptionRenderer.swift",
        "ProAppsCore/Files.swift",
        "AppleProApps/MeasurementService.swift",
        "AppleProApps/Program.swift",
    ):
        inputs.bind(native_sources / name)
    directory = inputs.path(str(directory.absolute()))
    if not directory.is_relative_to(Path.cwd() / "edits" / "work"):
        raise ValueError("Geometry candidates must stay under edits/work")
    directory.mkdir(mode=0o700)
    scratch, cache = directory / "scratch", directory / "cache"
    scratch.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    NativeCaptionLayout._save(directory / "attempt.json", {"inputs": inputs.hashes})
    geometry_path = directory / "geometry.json"
    geometry = audit.verify(paths["geometryConfig"], geometry_path)
    if geometry["geometryContained"] is not True:
        raise ValueError("Current static geometry is not contained")

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        execute_command(
            (
                "/usr/bin/env",
                f"TMPDIR={scratch}",
                f"TMP={scratch}",
                f"TEMP={scratch}",
                f"XDG_CACHE_HOME={cache}",
                *command,
            ),
            log,
            timeout,
        )

    layout = NativeCaptionLayout(
        executable=paths["native"],
        directory=directory / "native",
        style=style,
        width=canvas[0],
        height=canvas[1],
        evidence=(config, paths["fontAsset"], geometry_path),
        execute=execute,
    )
    layout.prepare_lines(texts)
    for text in texts:
        if text not in sizes or layout.measure(text) != sizes[text]:
            raise ValueError("Fresh native bounds differ from the calibration strings")
    for index in range(layout._batch):
        response = layout.directory / f"batch-{index:06d}.response.json"
        if envelope._font(inputs.json(response).get("font")) != font:
            raise ValueError("Fresh native font resolution differs from historical metrics")
    for name, digest in layout.verify_inputs().items():
        _reference(inputs, {"path": name, "sha256": digest})
    inputs.verify()
    receipt = directory / "scope.json"
    NativeCaptionLayout._save(
        receipt,
        {
            "version": 1,
            "scope": "static-geometry-candidate",
            "inputs": inputs.hashes,
            "measuredStrings": texts,
            "freshNativeFontMetricsMatched": True,
            "freshNativeBoundsMatched": True,
            "fontAssetResolutionVerified": False,
            "historicalTerminalOutputBound": False,
            "unseenGlyphsVerified": False,
            "colourApproved": False,
            "outlineApproved": False,
            "pixelGradientHoldApproved": False,
            "captionGeometryApproved": False,
            "deliveryComplete": False,
            "remaining": geometry["remaining"],
        },
    )
    # This is a new evidence-backed candidate, not an edit to the old template pin.
    result = directory / "envelope.json"
    evidence = tuple(
        dict.fromkeys(
            (
                receipt,
                geometry_path,
                paths["fontAsset"],
                *(Path(name) for name in layout.verify_inputs()),
            )
        )
    )
    if len(evidence) > envelope.MAX_EVIDENCE:
        raise ValueError("Geometry envelope evidence exceeds its budget")
    NativeCaptionLayout._save(
        result,
        {
            "version": 1,
            "canvas": list(canvas),
            "style": historical["style"],
            "nativeFont": historical["nativeFont"],
            "padding": historical["padding"],
            "template": {
                "path": geometry_paths["template"],
                "sha256": inputs.bind(inputs.path(geometry_paths["template"])),
            },
            "evidence": [{"path": str(path), "sha256": inputs.bind(path)} for path in evidence],
        },
    )
    calibrated = envelope.CalibratedMotionEnvelope(
        layout, calibration=result, calibration_sha256=inputs.bind(result), canvas=canvas
    )
    NativeCaptionLayout._save(
        directory / "envelopes.json",
        {
            "envelopes": [{"text": text, "bounds": asdict(calibrated(text))} for text in texts],
            "inputs": calibrated.verify_inputs(),
            "motionRenderVerified": False,
            "captionGeometryApproved": False,
            "deliveryComplete": False,
        },
    )
    inputs.verify()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    prepare(args.config, args.directory)


if __name__ == "__main__":
    main()
