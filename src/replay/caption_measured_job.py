"""Bounded measured caption candidates; never editor, render, or delivery approval."""

import argparse
import json
import os
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from time import monotonic

from replay import tracking_observations as storage
from replay.caption_io import _array, _fields, _number, _string
from replay.caption_measured_plan import MAX_TOKENS, measured_plan
from replay.caption_native_layout import NativeCaptionLayout
from replay.caption_source_support import (
    MAX_CUTS,
    MAX_WINDOWS,
    OutputSourceSupport,
    RetainedSourceSpan,
)
from replay.caption_timing import CaptionPolicy, SpokenToken
from replay.foreground_evidence import ForegroundWindow, _bound_bytes, read_window
from replay.foreground_geometry import _check_box
from replay.foreground_observations import _time
from replay.foreground_retained import verify as verify_retained
from replay.motion_native_envelope import CalibratedMotionEnvelope, _style
from replay.stages import execute_command, file_digest
from replay.vertical_captions import Box

ROOT = Path(__file__).resolve().parents[2]
MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_NATIVE_CALLS = 128
JOB_SECONDS = 300
CODE_FILES = (
    "caption_measured_job.py",
    "caption_measured_plan.py",
    "caption_source_support.py",
    "caption_native_layout.py",
    "motion_native_envelope.py",
    "caption_io.py",
    "caption_timing.py",
    "caption_display.py",
    "caption_linguistics.py",
    "foreground_evidence.py",
    "foreground_retained.py",
    "foreground_geometry.py",
    "vertical_captions.py",
    "subtitle_clock.py",
    "tracking_observations.py",
    "stages.py",
)


def _object(value: object, fields: str) -> dict[str, object]:
    raw = _fields(value)
    if set(raw) != set(fields.split()):
        raise ValueError("Unexpected or missing candidate fields")
    return raw


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _json(data: bytes) -> object:
    value: object = json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant)
    return value


def _local(value: object) -> Path:
    path = Path(_string(value))
    if not path.is_absolute():
        raise ValueError("Candidate paths must be absolute")
    path = storage._checked_output(path)
    if not path.is_relative_to(ROOT):
        raise ValueError("Candidate paths must stay inside the checkout")
    return path


def _merge(bindings: dict[str, str], additions: dict[str, str]) -> None:
    if any(bindings[name] != additions[name] for name in bindings.keys() & additions.keys()):
        raise ValueError("Conflicting candidate input bindings")
    bindings.update(additions)


def _pin(value: object, bindings: dict[str, str]) -> Path:
    raw = _object(value, "path sha256")
    path, digest = _local(raw["path"]), storage._digest(raw["sha256"])
    if not 0 < path.stat().st_size <= MAX_INPUT_BYTES:
        raise ValueError("Candidate input exceeds its byte budget")
    if file_digest(path) != digest:
        raise ValueError("Pinned candidate input changed")
    _merge(bindings, {str(path): digest})
    return path


def _document(value: object, bindings: dict[str, str]) -> object:
    path = _pin(value, bindings)
    return _json(_bound_bytes(path, bindings[str(path)], storage.CONFIG_BYTES))


def _rational(value: object) -> Fraction:
    return _time(_object(value, "value timescale"))


def _span(value: object) -> RetainedSourceSpan:
    raw = _object(value, "sourceStart outputStart duration")
    return RetainedSourceSpan(
        _rational(raw["sourceStart"]),
        _rational(raw["outputStart"]),
        _rational(raw["duration"]),
    )


def _token(value: object) -> SpokenToken:
    raw = _object(value, "text start end utterance")
    return SpokenToken(
        _string(raw["text"]),
        _number(raw["start"]),
        _number(raw["end"]),
        _string(raw["utterance"]),
    )


def _window(
    value: object,
    bindings: dict[str, str],
    verifications: list[dict[str, object]] | None = None,
) -> ForegroundWindow:
    fields = "config provenance"
    if "retainedVerification" in _fields(value):
        fields += " retainedVerification"
    raw = _object(value, fields)
    config = _pin(raw["config"], bindings)
    config_data = _fields(_document(raw["config"], bindings))
    # Original media may remain on its existing external volume. The pinned
    # reader verifies that exact source; only work/native inputs stay local.
    _local(config_data.get("nativeExecutable"))
    provenance = _pin(raw["provenance"], bindings)
    if provenance.name != "provenance.json":
        raise ValueError("Foreground reference must name provenance.json")
    if "retainedVerification" in raw:
        descriptor = _pin(raw["retainedVerification"], bindings)
        verified = verify_retained(
            config,
            provenance.parent,
            config_sha256=bindings[str(config)],
            provenance_sha256=bindings[str(provenance)],
            descriptor=descriptor,
            descriptor_sha256=bindings[str(descriptor)],
        )
        window = verified.window
        _merge(bindings, verified.inputs)
        if verifications is not None:
            verifications.append(verified.receipt())
    else:
        window = read_window(
            config,
            provenance.parent,
            config_sha256=bindings[str(config)],
            provenance_sha256=bindings[str(provenance)],
        )
    _merge(
        bindings,
        {
            str(window.job.source): window.job.source_hash,
            str(window.job.native): window.job.native_hash,
        },
    )
    return window


def plan(manifest: Path, directory: Path, *, manifest_sha256: str) -> Path:
    """Preserve attempted measurements and publish only a rechecked candidate.

    Run from the checkout root for the existing pinned morphology dictionary.
    Input pins establish byte identity, not correctness of clocks, cuts, semantic
    support, calibration, or correspondence with an actual editor timeline.
    """
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run candidate planning from the checkout root")
    directory = _local(str(directory.absolute()))
    if not directory.is_relative_to(ROOT / "edits" / "work"):
        raise ValueError("Candidate output must be under edits/work")
    bindings: dict[str, str] = {}
    manifest = _local(str(manifest.absolute()))
    raw = _object(
        _document({"path": str(manifest), "sha256": manifest_sha256}, bindings),
        "version tokens spans foreground native calibration outputFrameRate "
        "region centerX clearance",
    )
    if storage._integer(raw["version"]) != 1:
        raise ValueError("Unsupported candidate manifest version")
    token_data = _object(_document(raw["tokens"], bindings), "version clock tokens")
    if storage._integer(token_data["version"]) != 1 or token_data["clock"] != "output":
        raise ValueError("Tokens must explicitly use the output clock")
    tokens = tuple(_token(value) for value in _array(token_data["tokens"], MAX_TOKENS))
    if not tokens:
        raise ValueError("Candidate tokens cannot be empty")
    cuts = tuple(_span(value) for value in _array(raw["spans"], MAX_CUTS))
    references = _array(raw["foreground"], MAX_WINDOWS)
    verifications: list[dict[str, object]] = []
    windows = tuple(_window(value, bindings, verifications) for value in references)
    support = OutputSourceSupport(windows, cuts)
    canvas = (windows[0].job.width, windows[0].job.height)
    coordinates = _array(raw["region"], 4)
    if len(coordinates) != 4:
        raise ValueError("Region requires left, top, right, bottom")
    region = Box(*(_number(value) for value in coordinates))
    _check_box(region, relative=False)
    center, clearance = _number(raw["centerX"]), _number(raw["clearance"])
    if (
        region.right > canvas[0]
        or region.bottom > canvas[1]
        or not region.left < center < region.right
        or not 0 <= clearance <= canvas[1]
    ):
        raise ValueError("Region, center or clearance exceeds the canvas")
    policy = CaptionPolicy(
        frame_rate=storage._integer(raw["outputFrameRate"]), linguistic_boundaries=True
    )
    native = _pin(raw["native"], bindings)
    calibration = _pin(raw["calibration"], bindings)
    calibration_data = _fields(_document(raw["calibration"], bindings))
    _pin(calibration_data.get("template"), bindings)
    for reference in _array(calibration_data.get("evidence"), 32):
        _pin(reference, bindings)
    style = _style(calibration_data.get("style"))
    _merge(
        bindings,
        {
            str(Path(__file__).with_name(name).resolve()): file_digest(
                Path(__file__).with_name(name)
            )
            for name in CODE_FILES
        },
    )
    directory.mkdir(mode=0o700)
    if verifications:
        storage._write(directory / "retained-verification.json", {"windows": verifications})
    scratch, cache = directory / "scratch", directory / "cache"
    scratch.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    storage._write(
        directory / "attempt.json",
        {
            "manifest": raw,
            "inputs": bindings,
            "policy": asdict(policy),
            "maximumNativeCalls": MAX_NATIVE_CALLS,
            "planningBudgetSeconds": JOB_SECONDS,
            "candidateOnly": True,
            "deliveryComplete": False,
        },
    )
    deadline, calls = monotonic() + JOB_SECONDS, 0

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        nonlocal calls
        calls += 1
        remaining = int(deadline - monotonic())
        if calls > MAX_NATIVE_CALLS or remaining < 1:
            raise ValueError("Candidate native execution budget exhausted")
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
            min(timeout, remaining),
        )

    layout = NativeCaptionLayout(
        executable=native,
        directory=directory / "native",
        style=style,
        width=region.right - region.left,
        height=region.bottom - region.top,
        evidence=(manifest, calibration),
        execute=execute,
    )
    envelope = CalibratedMotionEnvelope(
        layout,
        calibration=calibration,
        calibration_sha256=bindings[str(calibration)],
        canvas=canvas,
    )
    candidates = measured_plan(
        tokens,
        policy=policy,
        region=region,
        center_x=center,
        clearance=clearance,
        output_support_top=support,
        envelope=envelope,
        prepare_lines=layout.prepare_lines,
    )
    _merge(bindings, envelope.verify_inputs())
    pending, result = directory / "candidate.pending.json", directory / "candidate.json"
    NativeCaptionLayout._save(
        pending,
        {
            "version": 1,
            "candidateOnly": True,
            "inputs": bindings,
            "canvas": canvas,
            "policy": asdict(policy),
            "coordinateSystem": "top-left canvas; calibrated text origin",
            "candidates": [asdict(candidate) for candidate in candidates],
            "editorCorrespondenceVerified": False,
            "completeMotionSupportVerified": False,
            "motionRenderVerified": False,
            "semanticPlacementApproved": False,
            "captionGeometryApproved": False,
            "aaaContrastApproved": False,
            "deliveryComplete": False,
        },
    )
    if tuple(_window(value, bindings) for value in references) != windows:
        raise ValueError("Foreground window changed during candidate planning")
    for name, digest in bindings.items():
        path = storage._checked_output(Path(name))
        if file_digest(path) != digest:
            raise ValueError("Candidate input bindings changed before publication")
    if monotonic() >= deadline:
        raise ValueError("Candidate planning deadline exceeded")
    # A partial JSON or failed final check never becomes the successful name.
    # Keep the pending file as evidence; linking refuses any existing destination.
    os.link(pending, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    plan(args.manifest, args.directory, manifest_sha256=args.manifest_sha256)


if __name__ == "__main__":
    main()
