"""Collect source-cell upper pixels in bounded native batches with pinned provenance."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import batched
from pathlib import Path

from replay import tracking_observations as storage
from replay.artifact_json import _pin
from replay.core import ReplayError
from replay.fcp_visible_arc import ArcCell, VisibleArc
from replay.foreground_observations import MAX_SAMPLES
from replay.head_image_plan import ImageFrameCell, image_cells
from replay.head_image_support import measured_upper_cells
from replay.recordings import record
from replay.stages import file_digest

RESPONSE_BYTES = 1_048_576
BATCH_TIMEOUT_SECONDS = 60
CODE_FILES = (
    "head_image_collection.py",
    "head_image_plan.py",
    "head_image_support.py",
    "foreground_observations.py",
    "fcp_visible_arc.py",
    "tracking_observations.py",
    "caption_style_defaults.py",
    "fcp_caption_srt.py",
    "fcp_project.py",
    "recordings.py",
    "stages.py",
    "core.py",
)


@dataclass(frozen=True, slots=True)
class ImageJob:
    config: Path
    source: Path
    native: Path
    sandbox: Path
    width: int
    height: int
    cells: tuple[ImageFrameCell, ...]
    inputs: dict[str, str]

    def request(self, cells: tuple[ImageFrameCell, ...]) -> dict[str, object]:
        if not cells or len(cells) > MAX_SAMPLES or any(cell.face is None for cell in cells):
            raise ReplayError("Native upper-support batches require observed same-frame faces")
        return {
            "path": str(self.source),
            "samples": [
                {"value": cell.start.numerator, "timescale": cell.start.denominator}
                for cell in cells
            ],
            "headFaces": [cell.request_face() for cell in cells],
        }


def load_job(config: Path) -> ImageJob:
    raw = record(json.loads(storage._read(config, storage.CONFIG_BYTES)))
    if (
        set(raw)
        != {
            "version",
            "source",
            "sourceSHA256",
            "nativeExecutable",
            "nativeSHA256",
            "sandbox",
            "sandboxSHA256",
            "width",
            "height",
            "cells",
            "inputs",
        }
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ReplayError("Invalid current-image collection configuration")
    source, native, sandbox = (
        storage._path(raw[key]) for key in ("source", "nativeExecutable", "sandbox")
    )
    width, height = storage._integer(raw["width"]), storage._integer(raw["height"])
    if not 1 <= width <= 16384 or not 1 <= height <= 16384 or width * height > 16_777_216:
        raise ReplayError("Current-image source geometry budget exceeded")
    raw_inputs = record(raw["inputs"])
    if not 1 <= len(raw_inputs) <= 128:
        raise ReplayError("Current-image input pin budget exceeded")
    inputs = {
        str(storage._path(path)): storage._digest(digest) for path, digest in raw_inputs.items()
    }
    for path, key in (
        (source, "sourceSHA256"),
        (native, "nativeSHA256"),
        (sandbox, "sandboxSHA256"),
    ):
        if inputs.get(str(path)) != storage._digest(raw[key]):
            raise ReplayError("Current-image source, native or sandbox pin differs")
    inputs[str(config.absolute())] = file_digest(config)
    inputs.update(
        {
            str(path): file_digest(path)
            for path in (Path(__file__).with_name(name) for name in CODE_FILES)
        }
    )
    return ImageJob(
        config.absolute(),
        source,
        native,
        sandbox,
        width,
        height,
        image_cells(raw["cells"], width=width, height=height),
        inputs,
    )


def _bindings(job: ImageJob) -> dict[str, str]:
    if not os.access(job.native, os.X_OK):
        raise ReplayError("Current-image native executable is not executable")
    actual = {path: file_digest(Path(path)) for path in job.inputs}
    if actual != job.inputs:
        raise ReplayError("Current-image producer input changed")
    return actual


def _batches(job: ImageJob) -> tuple[tuple[ImageFrameCell, ...], ...]:
    return tuple(batched((cell for cell in job.cells if cell.face is not None), MAX_SAMPLES))


def _batch_cells(
    job: ImageJob, directory: Path, cells: tuple[ImageFrameCell, ...]
) -> tuple[ArcCell, ...]:
    request, response = directory / "request.json", directory / "response.json"
    if json.loads(storage._read(request, storage.CONFIG_BYTES)) != job.request(cells):
        raise ReplayError("Current-image request differs from its actual source cells")
    storage._check_diagnostics(directory)
    return measured_upper_cells(
        json.loads(storage._read(response, RESPONSE_BYTES)),
        times=tuple(cell.start for cell in cells),
        frame_ends=tuple(cell.end for cell in cells),
        width=job.width,
        height=job.height,
    )


def _receipt(job: ImageJob, output: Path) -> tuple[dict[str, object], VisibleArc]:
    measured: dict[Fraction, ArcCell] = {}
    inputs = dict(job.inputs)
    batches: list[dict[str, object]] = []
    for index, cells in enumerate(_batches(job)):
        directory = output / f"batch-{index:04d}"
        for cell in _batch_cells(job, directory, cells):
            measured[cell.start] = cell
        paths = (directory / "request.json", directory / "response.json")
        pins = {str(path): file_digest(path) for path in paths}
        inputs.update(pins)
        batches.append({"inputs": pins, "frames": len(cells)})
    arc_cells = tuple(
        measured.get(cell.start, ArcCell(cell.start, cell.end, None, 0, 0)) for cell in job.cells
    )
    receipt: dict[str, object] = {
        "version": 1,
        "inputs": job.inputs,
        "batches": batches,
        "cells": [
            {
                "start": str(cell.start),
                "end": str(cell.end),
                "upperPixelY": cell.visible_top,
                "supportedPixels": cell.supported,
                "unknownPixels": cell.unknown,
            }
            for cell in arc_cells
        ],
        "frames": len(arc_cells),
        "measuredFrames": len(measured),
        "supportedFrames": sum(cell.visible_top is not None for cell in arc_cells),
        "nativeExitCode": 0,
        "candidateOnly": True,
        "semanticHeadVerified": False,
        "captionPlacementApproved": False,
    }
    return receipt, VisibleArc(arc_cells, inputs)


def observe(config: Path, output: Path) -> None:
    """No raster retention, sparse interpolation, missing-face holds or silent retries."""
    job = load_job(config)
    _bindings(job)
    output = storage._checked_output(output)
    output.mkdir(mode=0o700)
    storage._write(output / "inputs.json", job.inputs)
    for index, cells in enumerate(_batches(job)):
        directory = output / f"batch-{index:04d}"
        directory.mkdir(mode=0o700)
        request = directory / "request.json"
        storage._write(request, job.request(cells))
        with (
            (directory / "response.json").open("xb") as stdout,
            (directory / "stderr.log").open("xb") as stderr,
        ):
            os.fchmod(stdout.fileno(), 0o600)
            os.fchmod(stderr.fileno(), 0o600)
            subprocess.run(
                (
                    "/usr/bin/sandbox-exec",
                    "-f",
                    str(job.sandbox),
                    str(job.native),
                    "measure-media",
                    "video_foreground_measure",
                    str(request),
                ),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=BATCH_TIMEOUT_SECONDS,
                check=True,
            )
        _batch_cells(job, directory, cells)
        print(f"UPPER_PIXEL_BATCH {index}: {len(cells)} actual source cells", flush=True)
    receipt, _arc = _receipt(job, output)
    _bindings(job)
    storage._write(output / "provenance.json", receipt)


def read_arc(config: Path, output: Path) -> VisibleArc:
    """Verify every raw response before returning current approximate image support."""
    job = load_job(config)
    _bindings(job)
    output = storage._checked_output(output)
    expected, arc = _receipt(job, output)
    recorded = json.loads(storage._read(output / "provenance.json", storage.CONFIG_BYTES))
    stored_inputs = json.loads(storage._read(output / "inputs.json", storage.CONFIG_BYTES))
    if stored_inputs != job.inputs or json.dumps(recorded, sort_keys=True) != json.dumps(
        expected, sort_keys=True
    ):
        raise ReplayError(
            "Current-image provenance differs from its actual source-cell measurements"
        )
    _bindings(job)
    arc.inputs.update(
        {
            str(path): file_digest(path)
            for path in (output / "inputs.json", output / "provenance.json")
        }
    )
    return arc


def workflow(config: Path, evidence: Path, output: Path) -> None:
    """Declare collection plus its domain verifier for the normal workflow preflight."""
    job = load_job(config)
    _bindings(job)
    evidence = storage._checked_output(evidence)
    command = (sys.executable, "-m", "replay.head_image_collection")
    storage._write(
        storage._checked_output(output),
        {
            "version": 1,
            "stages": [
                {
                    "name": "current-upper-pixels",
                    "command": [*command, "observe", str(job.config), str(evidence)],
                    "verify_command": [*command, "verify", str(job.config), str(evidence)],
                    "inputs": list(job.inputs),
                    "outputs": [str(evidence / "inputs.json"), str(evidence / "provenance.json")],
                    "timeout_seconds": BATCH_TIMEOUT_SECONDS * 16 + 300,
                }
            ],
        },
    )


def read_image_binding(value: object, *, source_hash: str, width: int, height: int) -> VisibleArc:
    """Bind the caller's verified original source and coordinate canvas to raw pixels."""
    binding = record(value)
    if set(binding) != {"config", "provenance"}:
        raise ReplayError("Current-image binding requires its pinned config and provenance")
    config, provenance = _pin(binding["config"]), _pin(binding["provenance"])
    job = load_job(config)
    if (
        provenance.name != "provenance.json"
        or job.inputs[str(job.source)] != source_hash
        or (job.width, job.height) != (width, height)
    ):
        raise ReplayError("Current-image binding differs from the source or coordinate geometry")
    return read_arc(config, provenance.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("observe", "verify", "workflow"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.action == "observe":
        observe(args.config, args.output)
    elif args.action == "verify":
        arc = read_arc(args.config, args.output)
        print(f"Verified {len(arc.cells)} current source cells; semantic head remains unverified")
    else:
        if args.evidence is None:
            parser.error("workflow requires --evidence")
        workflow(args.config, args.evidence, args.output)


if __name__ == "__main__":
    main()
