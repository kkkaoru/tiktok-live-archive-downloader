"""Per-frame caption boxes from yolo-track's single detection pass, streamed for the cleanup.

The source-caption cleanup used to decode every chunk and run the same detector
(``subtitle_yolo_regions``) that yolo-track had already run over the whole video. yolo-track now
detects every frame once and ``Writer`` turns each prediction into exactly what a cleanup chunk
computes: caption-class boxes grown along their glyphs in the detector frame
(``extend_to_glyphs``), then scaled to canvas pixels. Every ``RANGE_FRAMES`` frames one range
file is published atomically, keyed by the decoded presentation clock of the clocks step (frame
``n`` of the whole-file decode from frame 0 is that clock's ``n``-th timestamp; a per-range time
seek would shift frames). ``complete.json`` marks the end of the pass.

A chunk reads its boxes with ``load`` only when every one of its timestamps is present and the
files name the same source, weights, detector size and clock time base; otherwise it detects as
before. Equal decoded pixels and batch-invariant predictions make the two paths give the same
boxes (shown on 900-frame chunks for batch 16/32 and both NMS variants, 2026-10-24).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay import subtitle_yolo_regions as regions
from replay import yolo_track
from replay.core import ReplayError
from replay.recordings import record

PROTOCOL = "yolo-caption-boxes-v1"
RANGE_FRAMES = 900
COMPLETE = "complete.json"
RANGE = re.compile(r"^range-(?P<first>[0-9]{12})-(?P<last>[0-9]{12})\.json$")
MAX_RANGE_BYTES = 64 * 1024 * 1024

Boxes = list[list[regions.Corners]]


def frame_clock(path: Path) -> tuple[tuple[int, ...], Fraction]:
    """The decoded clock (``video-clock.json``: ffprobe frames and the stream time base)."""
    data = record(json.loads(path.read_text(encoding="utf-8")))
    frames, streams = data.get("frames"), data.get("streams")
    if not isinstance(frames, list) or not frames or not isinstance(streams, list) or not streams:
        raise ReplayError(f"{path} is not a decoded video clock")
    pts = tuple(int(str(record(frame)["pts"])) for frame in frames)
    return pts, Fraction(str(record(streams[0])["time_base"]))


def _write(path: Path, value: object) -> None:
    staged = path.with_name(f".{path.name}.{os.getpid()}.partial")
    staged.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    staged.replace(path)  # a reader sees a whole range file or none


class Writer:
    """``yolo_track`` sink: every batch's caption boxes, published in range files."""

    def __init__(
        self,
        directory: Path,
        *,
        clock_path: Path,
        source: Path,
        weights_sha256: str,
        imgsz: int,
        ffprobe: str,
    ) -> None:
        self.directory = directory
        self.pts, self.time_base = frame_clock(clock_path)
        width, _, _ = yolo_track.probe_video(source, ffprobe)
        self.scale = width / yolo_track.TRAIN_WIDTH
        self.identity = {
            "protocol": PROTOCOL,
            "source": yolo_track.sha256(source),
            "weights": weights_sha256,
            "imgsz": imgsz,
            "confidence": yolo_track.CONFIDENCE_FLOOR,
            "timeBase": str(self.time_base),
        }
        self.pending: list[tuple[int, list[list[float]]]] = []
        self.frames = 0
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        (directory / COMPLETE).unlink(missing_ok=True)  # a new pass: not complete until it ends

    def __call__(
        self,
        frames: Sequence[tuple[int, NDArray[np.uint8]]],
        results: Sequence[yolo_track.ResultLike],
    ) -> None:
        for (number, image), boxes in zip(frames, regions.caption_boxes(results), strict=True):
            grown = [regions.extend_to_glyphs(image, box) for box in boxes]
            self.pending.append((number, [[value * self.scale for value in box] for box in grown]))
            if len(self.pending) == RANGE_FRAMES:
                self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        numbers = [number for number, _ in self.pending]
        if numbers[-1] >= len(self.pts) or numbers != list(range(numbers[0], numbers[-1] + 1)):
            raise ReplayError("Detected frames do not follow the decoded clock")
        pts = [self.pts[number] for number in numbers]
        name = f"range-{pts[0]:012d}-{pts[-1]:012d}.json"
        boxes = [frame for _, frame in self.pending]
        _write(self.directory / name, {**self.identity, "pts": pts, "boxes": boxes})
        self.frames += len(numbers)
        self.pending.clear()

    def close(self) -> None:
        """Publish the last range and mark the pass complete."""
        self.flush()
        _write(self.directory / COMPLETE, {**self.identity, "frames": self.frames})


def complete(directory: Path) -> bool:
    return (directory / COMPLETE).is_file()


def _ranges(directory: Path) -> list[tuple[int, int, Path]]:
    if not directory.is_dir():
        return []
    found = (RANGE.match(path.name) for path in directory.iterdir())
    return sorted(
        (int(match["first"]), int(match["last"]), directory / match.string)
        for match in found
        if match
    )


def covered(directory: Path, first: int, last: int) -> bool:
    """True once range files reach from ``first`` to ``last`` (checked exactly by ``load``)."""
    ranges = [(low, high) for low, high, _ in _ranges(directory) if high >= first and low <= last]
    return bool(ranges) and ranges[0][0] <= first and ranges[-1][1] >= last


def load(
    directory: Path,
    pts: Sequence[int],
    *,
    time_base: Fraction,
    source_sha256: str,
    weights_sha256: str,
    imgsz: int,
) -> Boxes | None:
    """The stored boxes of exactly these frames, or None when any is missing or differs."""
    if not pts or not covered(directory, pts[0], pts[-1]):
        return None
    expected = {
        "protocol": PROTOCOL,
        "source": source_sha256,
        "weights": weights_sha256,
        "imgsz": imgsz,
        "confidence": regions.CONFIDENCE,
        "timeBase": str(time_base),
    }
    stored: dict[int, list[regions.Corners]] = {}
    for low, high, path in _ranges(directory):
        if high < pts[0] or low > pts[-1]:
            continue
        if path.stat().st_size > MAX_RANGE_BYTES:
            raise ReplayError(f"{path.name} exceeds its byte budget")
        data = record(json.loads(path.read_text(encoding="utf-8")))
        if {key: data.get(key) for key in expected} != expected:
            return None
        times, boxes = data.get("pts"), data.get("boxes")
        if not isinstance(times, list) or not isinstance(boxes, list) or len(times) != len(boxes):
            raise ReplayError(f"{path.name} is malformed")
        for time, frame in zip(times, boxes, strict=True):
            stored[int(time)] = [
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])) for box in frame
            ]
    if any(time not in stored for time in pts):
        return None
    return [stored[time] for time in pts]
