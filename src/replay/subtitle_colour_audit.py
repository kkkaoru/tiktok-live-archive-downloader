"""Decode audited frames of a cleaned candidate and its source, and count residual caption colour.

Both videos are read through ffmpeg rawvideo pipes in lockstep (one frame resident per video).
The original's recognised line boxes per frame bound the residual count so scene pixels with the
caption hue elsewhere are ignored. Frame numbers are decode order within each video.
"""

import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from replay.subtitle_colour_residual import unchanged_pink
from replay.subtitle_colour_support import Box, box_mask, pink
from replay.subtitle_inpaint import Image

CHANNELS = 3
MINIMUM_CAPTION_PINK = 40
PROCESS_SECONDS = 1800


def _selector(indices: Sequence[int]) -> str:
    """Compact ffmpeg ``select`` expression: one term per arithmetic run of frame numbers."""
    terms: list[str] = []
    position = 0
    while position < len(indices):
        stride = indices[position + 1] - indices[position] if position + 1 < len(indices) else 1
        end = position
        while end + 1 < len(indices) and indices[end + 1] - indices[end] == stride:
            end += 1
        first, last = indices[position], indices[end]
        if end == position:
            terms.append(f"eq(n\\,{first})")
        else:
            terms.append(f"between(n\\,{first}\\,{last})*not(mod(n-{first}\\,{stride}))")
        position = end + 1
    return "+".join(terms)


def _frames(
    ffmpeg: str, video: Path, start: float, indices: Sequence[int], size: tuple[int, int]
) -> Iterator[Image]:
    """Yield the requested decode-order frames (ascending, unique) as BGR arrays."""
    wanted = _selector(indices)
    command = [ffmpeg, "-nostdin", "-v", "error"]
    command += ["-ss", f"{start:.6f}"] if start else []
    command += ["-i", str(video), "-vf", f"select='{wanted}'", "-fps_mode", "passthrough"]
    command += ["-frames:v", str(len(indices)), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    width, height = size
    length = width * height * CHANNELS
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        stream = process.stdout
        if stream is None:
            raise ValueError("ffmpeg produced no output pipe")
        for _ in indices:
            raw = stream.read(length)
            if len(raw) != length:
                raise ValueError("Decoder ended before every audited frame")
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, CHANNELS)
        process.wait(timeout=PROCESS_SECONDS)
        if process.returncode:
            raise ValueError("ffmpeg failed while decoding audited frames")


def colour_counts(
    *,
    ffmpeg: str,
    source: Path,
    source_start: float,
    cleaned: Path,
    indices: Sequence[int],
    boxes: Sequence[Sequence[Box]],
    size: tuple[int, int],
) -> list[int]:
    """Unchanged caption-colour pixels per audited frame (``indices`` ascending, unique)."""
    if list(indices) != sorted(set(indices)) or not indices:
        raise ValueError("Audited frame indices must be ascending and unique")
    originals = _frames(ffmpeg, source, source_start, indices, size)
    repaired = _frames(ffmpeg, cleaned, 0.0, indices, size)
    counts: list[int] = []
    for index, original, clean in zip(indices, originals, repaired, strict=True):
        allowed = box_mask((size[1], size[0]), (0, 0), boxes[index])
        counts.append(unchanged_pink(clean, original, allowed))
    return counts


def source_pink_counts(
    *,
    ffmpeg: str,
    source: Path,
    source_start: float,
    lines: Sequence[tuple[int, tuple[float, float, float, float]]],
    size: tuple[int, int],
) -> list[int]:
    """Caption-colour pixels of the SOURCE frame inside each (frame, box) line, in input order.

    A line recognised in a repaired frame that holds no caption-coloured source pixels was never
    caption text (interface text behind the caption, for example); one that does stays a residual.
    """
    if not lines:
        raise ValueError("No lines to classify")
    indices = sorted({frame for frame, _ in lines})
    width, height = size
    pink_by_frame = {
        index: pink(frame)
        for index, frame in zip(
            indices, _frames(ffmpeg, source, source_start, indices, size), strict=True
        )
    }
    counts: list[int] = []
    for frame, (left, top, box_width, box_height) in lines:
        x0, y0 = max(0, int(left)), max(0, int(top))
        x1, y1 = min(width, int(left + box_width) + 1), min(height, int(top + box_height) + 1)
        counts.append(int(pink_by_frame[frame][y0:y1, x0:x1].sum()))
    return counts
