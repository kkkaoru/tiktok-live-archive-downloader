"""Adapt validated sampled-OCR chunks to the cleanup-assembly consumer layout.

``subtitle_cleanup_assembly.prepare`` reads ``chunk.json``, ``candidate-config.json`` and
``repair/candidate.json`` per chunk beside a ``plan.json``. A sampled chunk keeps its own
receipt and SSD media, so this writes only hash-bound metadata that points at the accepted
attempt's media (no copy). Every chunk must first pass the sampled eligibility validator.
Quality stays unapproved; the consumer's own media and decoded-clock verifier still runs.
"""

import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.local_asr import fingerprint
from replay.recordings import record
from replay.subtitle_cleanup_plan import clock_chunks
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_sampled_eligibility import validate_sampled_eligibility

MEDIA_NAME = "candidate.mp4"


@dataclass(frozen=True, slots=True)
class Adaptation:
    """Where sampled chunk evidence lives and where the consumer layout is written."""

    configs: Path
    states: Path
    settings: Path
    output: Path
    media_root: Path
    chunk_frames: int
    # The pinned clock moved with its work root: use this copy when its hash equals the pin.
    source_clock: Path | None = None


def _read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": fingerprint(path)}


def _clock(config: dict[str, object]) -> VideoFrameClock:
    data = record(config.get("clock"))
    pts, base, last = data.get("pts"), data.get("timeBase"), data.get("lastDuration")
    if not isinstance(pts, list) or not isinstance(base, str) or type(last) is not int:
        raise ValueError("Chunk configuration lacks an exact source clock")
    if any(type(item) is not int for item in pts):
        raise ValueError("Chunk clock contains non-integer PTS")
    return VideoFrameClock(tuple(item for item in pts if type(item) is int), Fraction(base), last)


def _text_map(value: object) -> dict[str, str]:
    data = record(value)
    if any(not isinstance(item, str) for item in data.values()):
        raise ValueError("Receipt pins must be text")
    return {key: str(item) for key, item in data.items()}


def _settings(adaptation: Adaptation, native: Path, native_sha: str, clock: Path) -> dict[str, str]:
    """Write the base settings with the sampled-run native and the verified clock location."""
    base = _read(adaptation.settings)
    base["native"] = {"path": str(native), "sha256": native_sha}
    base["sourceClock"] = _pin(clock)
    base["sampledRepair"] = {"basedOn": _pin(adaptation.settings), "qualityApproved": False}
    target = adaptation.output / "settings.json"
    _write(target, base)
    return _pin(target)


def _identity(adaptation: Adaptation) -> tuple[str, Path, str]:
    """Return the source hash and release native shared by all chunk configurations."""
    config = _read(adaptation.configs / "chunk-00000.json")
    receipt = _read(adaptation.states / "chunk-00000" / "chunk-receipt.json")
    pins = _text_map(receipt.get("pins"))
    native = Path(str(config.get("native")))
    if fingerprint(native) != pins.get("native") or fingerprint(Path(str(config["source"]))) != (
        pins.get("source")
    ):
        raise ValueError("Source or native media differs from the receipt pins")
    return pins["source"], native, pins["native"]


def _media(config: dict[str, object], accepted: str, root: Path) -> Path:
    media = Path(str(config.get("mediaOutput"))) / accepted / MEDIA_NAME
    if not media.is_file() or media.is_symlink() or not media.is_relative_to(root):
        raise ValueError("Accepted attempt media is missing or outside the media root")
    return media


def _chunk(
    index: int, clock: VideoFrameClock, adaptation: Adaptation, settings: dict[str, str]
) -> None:
    """Validate one sampled chunk and write its consumer-layout metadata."""
    config = _read(adaptation.configs / f"chunk-{index:05d}.json")
    state = adaptation.states / f"chunk-{index:05d}"
    receipt = _read(state / "chunk-receipt.json")
    accepted = str(receipt.get("accepted"))
    admission = _read(state / accepted / "admission.json")
    pins = _text_map(receipt.get("pins"))
    width, height = config.get("width"), config.get("height")
    if _clock(config) != clock or type(width) is not int or type(height) is not int:
        raise ValueError(f"Chunk {index} clock or canvas differs from the source plan")
    validate_sampled_eligibility(
        receipt,
        admission,
        source_sha256=pins.get("source", ""),
        clock=clock,
        width=width,
        height=height,
        pins=pins,
    )
    media = _media(config, accepted, adaptation.media_root)
    directory = adaptation.output / f"chunk-{index:05d}"
    (directory / "repair").mkdir(mode=0o700, parents=True)
    clock_data = record(config["clock"])
    receipt_pin = _pin(state / "chunk-receipt.json")
    _write(
        directory / "chunk.json",
        {
            "version": 1,
            "settings": settings,
            "clock": clock_data,
            "ocrDirectory": str(directory / "ocr"),
            "sampledReceipt": receipt_pin,
        },
    )
    producer = {
        "source": config["source"],
        "sourceSHA256": pins["source"],
        "nativeExecutable": config["native"],
        "nativeSHA256": pins["native"],
        "outputDirectory": str(directory / "repair"),
        "width": width,
        "height": height,
        "clock": clock_data,
        "mediaOutput": str(media),
        "allowedMediaRoot": str(adaptation.media_root),
        "sampledReceipt": receipt_pin,
        "sampledAdmission": _pin(state / accepted / "admission.json"),
        "sampledEligibilityValidated": True,
    }
    _write(directory / "candidate-config.json", producer)
    _write(
        directory / "repair" / "candidate.json",
        {
            "inputs": {
                str(directory / "candidate-config.json"): fingerprint(
                    directory / "candidate-config.json"
                ),
                str(config["source"]): pins["source"],
                str(config["native"]): pins["native"],
            },
            "render": {"frames": len(clock.pts)},
            "sourceStart": str(clock.pts[0] * clock.time_base),
            "videoSHA256": fingerprint(media),
            "mediaOutput": str(media),
            "sampledReceipt": receipt_pin,
            "qualityApproval": False,
            "fullDecodeVerified": False,
            "deliveryComplete": False,
        },
    )


def adapt(adaptation: Adaptation) -> Path:
    """Write the consumer layout for every chunk of the source plan; return ``plan.json``."""
    adaptation.output.mkdir(mode=0o700, parents=True)
    source_sha, native, native_sha = _identity(adaptation)
    clock_ref = record(_read(adaptation.settings).get("sourceClock"))
    clock_path = adaptation.source_clock or Path(str(clock_ref.get("path")))
    if fingerprint(clock_path) != clock_ref.get("sha256"):
        raise ValueError("Source clock differs from the settings pin")
    settings = _settings(adaptation, native, native_sha, clock_path)
    source_clock = _read(clock_path)
    clocks = clock_chunks(source_clock, maximum_frames=adaptation.chunk_frames)
    for index, clock in enumerate(clocks):
        _chunk(index, clock, adaptation, settings)
    frames = sum(len(clock.pts) for clock in clocks)
    plan = adaptation.output / "plan.json"
    _write(
        plan,
        {
            "version": 1,
            "settings": settings,
            "totalSourceFrames": frames,
            "plannedFrames": frames,
            "chunks": len(clocks),
            "wholeSourcePlanned": True,
            "sourceSHA256": source_sha,
            "qualityApproved": False,
        },
    )
    return plan
