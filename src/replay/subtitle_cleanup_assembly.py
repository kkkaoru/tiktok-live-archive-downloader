"""Prepare pinned source assembly inputs from complete, exact-clock cleanup chunks.

This metadata-only preparation never approves repair quality, hashes media, or
replaces the mandatory delivery_source_assembly packet/decoded-clock verifier.
It reads existing results without starting OCR, repair, or encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

from replay.delivery_source_assembly import candidate_media_path
from replay.recordings import is_record
from replay.subtitle_cleanup_plan import clock_chunks
from replay.subtitle_clock import VideoFrameClock

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CHUNKS = 512
CODE_FILES = (
    "delivery_source_assembly.py",
    "subtitle_cleanup_assembly.py",
    "subtitle_cleanup_plan.py",
    "subtitle_clock.py",
    "recordings.py",
    "caption_io.py",
    "caption_masks.py",
    "subtitle_plan.py",
    "subtitle_profile.py",
    "core.py",
)


def _local(path: Path) -> Path:
    root = Path.cwd().resolve()
    if not path.is_absolute() or path != path.resolve() or not path.is_relative_to(root):
        raise ValueError("Assembly preparation paths must be canonical checkout-local paths")
    return path


def _object(value: object) -> dict[str, object]:
    if not is_record(value):
        raise ValueError("Expected a JSON object")
    return value


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected an exact SHA256 pin")
    return value


def _name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(char in value for char in "\n\r\x00")
        or not Path(value).is_absolute()
        or Path(value) != Path(value).resolve()
    ):
        raise ValueError("Expected a canonical absolute reference")
    return value


def _bytes(path: Path, *, code: bool = False) -> bytes:
    _local(path)
    if not code and not path.is_relative_to(Path.cwd().resolve() / "edits/work"):
        raise ValueError("Preparation metadata must remain in checkout-local work")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_JSON_BYTES:
            raise ValueError("Missing or excessive bounded metadata")
        data = stream.read(MAX_JSON_BYTES + 1)
        after = os.fstat(stream.fileno())
    old = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    new = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if old != new or len(data) != before.st_size:
        raise ValueError("Metadata changed during reading")
    return data


def _read(path: Path, inputs: dict[str, str]) -> dict[str, object]:
    data = _bytes(path)
    digest = hashlib.sha256(data).hexdigest()
    if inputs.get(str(path), digest) != digest:
        raise ValueError("Metadata fingerprint changed")
    inputs[str(path)] = digest
    return _object(json.loads(data, object_pairs_hook=_unique))


def _ref(value: object, inputs: dict[str, str]) -> tuple[Path, dict[str, object]]:
    ref = _object(value)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Expected a pinned metadata reference")
    path = _local(Path(_name(ref["path"])))
    data = _read(path, inputs)
    if inputs[str(path)] != _sha(ref["sha256"]):
        raise ValueError("Metadata reference pin changed")
    return path, data


def _identity(value: object) -> tuple[str, str]:
    ref = _object(value)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Expected a pinned source/native identity")
    return _name(ref["path"]), _sha(ref["sha256"])


def _positive(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("Expected a positive integer")
    return value


def _clock_data(clock: VideoFrameClock) -> dict[str, object]:
    return {
        "pts": list(clock.pts),
        "timeBase": str(clock.time_base),
        "lastDuration": clock.last_duration,
    }


def _chunk(
    directory: Path,
    clock: VideoFrameClock,
    settings: Path,
    source: tuple[str, str],
    native: tuple[str, str],
    inputs: dict[str, str],
) -> tuple[dict[str, object], dict[str, object], tuple[int, int]]:
    chunk = _read(directory / "chunk.json", inputs)
    producer_path, receipt_path = (
        directory / "candidate-config.json",
        directory / "repair/candidate.json",
    )
    producer, receipt = _read(producer_path, inputs), _read(receipt_path, inputs)
    expected_clock = _clock_data(clock)
    if (
        type(chunk.get("version")) is not int
        or chunk["version"] != 1
        or chunk.get("clock") != expected_clock
        or chunk.get("settings") != {"path": str(settings), "sha256": inputs[str(settings)]}
        or chunk.get("ocrDirectory") != str(directory / "ocr")
        or producer.get("clock") != expected_clock
        or producer.get("source") != source[0]
        or producer.get("sourceSHA256") != source[1]
        or producer.get("nativeExecutable") != native[0]
        or producer.get("nativeSHA256") != native[1]
        or producer.get("outputDirectory") != str(directory / "repair")
    ):
        raise ValueError("Chunk producer/settings/source/native/clock identity differs")
    width, height = _positive(producer.get("width")), _positive(producer.get("height"))
    if max(width, height) > 16384:
        raise ValueError("Excessive chunk geometry")
    bound = _object(receipt.get("inputs"))
    if (
        bound.get(str(producer_path)) != inputs[str(producer_path)]
        or bound.get(source[0]) != source[1]
        or bound.get(native[0]) != native[1]
        or _positive(_object(receipt.get("render")).get("frames")) != len(clock.pts)
        or receipt.get("sourceStart") != str(clock.pts[0] * clock.time_base)
    ):
        raise ValueError("Incomplete or differently bound cleanup result")
    _sha(receipt.get("videoSHA256"))
    media = candidate_media_path(producer)
    if "mediaOutput" in producer and receipt.get("mediaOutput") != str(media):
        raise ValueError("Cleanup receipt differs from its explicit media destination")
    if not media.is_file() or media.stat().st_size <= 0:
        raise ValueError("Completed cleanup result requires existing media")
    row = {
        "media": str(media),
        "sourceFirstPTS": clock.pts[0],
        "sourceEndPTS": clock.pts[-1] + clock.last_duration,
        "frames": len(clock.pts),
        "timeBase": str(clock.time_base),
    }
    refs: dict[str, object] = {
        "producer": {"path": str(producer_path), "sha256": inputs[str(producer_path)]},
        "candidate": {"path": str(receipt_path), "sha256": inputs[str(receipt_path)]},
    }
    return row, refs, (width, height)


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(plan: Path, output: Path) -> Path:
    """Require every chunk before exclusively publishing a ready-to-verify manifest."""
    _local(plan)
    _local(output)
    if not output.is_relative_to(Path.cwd().resolve() / "edits/work"):
        raise ValueError("Preparation outputs must remain in checkout-local work")
    inputs: dict[str, str] = {}
    raw = _read(plan, inputs)
    settings_path, settings = _ref(raw.get("settings"), inputs)
    source, native = _identity(settings.get("source")), _identity(settings.get("native"))
    clock_path, source_clock = _ref(settings.get("sourceClock"), inputs)
    clocks = clock_chunks(source_clock, maximum_frames=_positive(settings.get("chunkFrames")))
    frames = sum(len(clock.pts) for clock in clocks)
    if (
        type(raw.get("version")) is not int
        or raw["version"] != 1
        or type(settings.get("version")) is not int
        or settings["version"] != 1
        or raw.get("wholeSourcePlanned") is not True
        or _positive(raw.get("chunks")) != len(clocks)
        or len(clocks) > MAX_CHUNKS
        or _positive(raw.get("totalSourceFrames")) != frames
        or _positive(raw.get("plannedFrames")) != frames
        or clocks[0].pts[0] != 0
    ):
        raise ValueError("Assembly requires the complete zero-origin physical source plan")
    rows: list[dict[str, object]] = []
    refs: list[dict[str, object]] = []
    canvases: set[tuple[int, int]] = set()
    for index, clock in enumerate(clocks):
        row, ref, canvas = _chunk(
            plan.parent / f"chunk-{index:05d}", clock, settings_path, source, native, inputs
        )
        rows.append(row)
        refs.append(ref)
        canvases.add(canvas)
    if len(canvases) != 1:
        raise ValueError("Cleanup chunk geometry differs")
    code: dict[str, str] = {}
    for name in CODE_FILES:
        path = Path(__file__).with_name(name).resolve()
        digest = hashlib.sha256(_bytes(path, code=True)).hexdigest()
        inputs[str(path)] = digest
        code[str(path)] = digest
    for name, digest in inputs.items():
        if hashlib.sha256(_bytes(Path(name), code=name in code)).hexdigest() != digest:
            raise ValueError("Preparation input changed before publication")
    targets = [output / name for name in ("index.json", "manifest.json", "preparation.json")]
    if not output.is_dir() or any(path.exists() or path.is_symlink() for path in targets):
        raise ValueError("Preparation outputs must be absent in an existing directory")
    _write(
        targets[0],
        {"purpose": "ordered source-clock cleanup candidates; quality unapproved", "ranges": rows},
    )
    index_sha = hashlib.sha256(_bytes(targets[0])).hexdigest()
    _write(
        targets[1],
        {"version": 1, "index": {"path": str(targets[0]), "sha256": index_sha}, "chunks": refs},
    )
    _write(
        targets[2],
        {
            "version": 1,
            "inputs": inputs,
            "currentPreparationCode": code,
            "sourceIdentity": {"path": source[0], "sha256": source[1]},
            "sourceClock": {"path": str(clock_path), "sha256": inputs[str(clock_path)]},
            "chunks": len(clocks),
            "physicalSourceFrames": frames,
            "physicalSourceTerminalPTS": clocks[-1].pts[-1] + clocks[-1].last_duration,
            "timeBase": str(clocks[0].time_base),
            "canvas": list(canvases.pop()),
            "manifest": {
                "path": str(targets[1]),
                "sha256": hashlib.sha256(_bytes(targets[1])).hexdigest(),
            },
            "exactPreparedSourceClockVerified": True,
            "candidateOnly": True,
            "requiredNextStage": (
                "delivery_source_assembly full media, packets and decoded PTS verification"
            ),
            "mediaBytesVerified": False,
            "sourceCaptionQualityApproved": False,
            "deliveryComplete": False,
        },
    )
    return targets[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(prepare(args.plan, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
