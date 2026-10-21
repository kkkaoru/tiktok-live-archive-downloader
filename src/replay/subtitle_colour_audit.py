"""Decode audited frames of a cleaned candidate and its source, and count residual caption colour.

Both videos are read through ffmpeg rawvideo pipes in lockstep (one frame resident per video).
The original's recognised line boxes per frame bound the residual count so scene pixels with the
caption hue elsewhere are ignored. Frame numbers are decode order within each video.
"""

import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from replay.subtitle_colour_residual import SCENE_KERNEL, unchanged_pink
from replay.subtitle_colour_support import BOX_MARGIN, Box, box_mask, pink
from replay.subtitle_inpaint import Image

CHANNELS = 3
MINIMUM_CAPTION_PINK = 40
PROCESS_SECONDS = 1800
# Rows around a box that a count reads: the box margin plus the scene-object opening's reach.
REACH = BOX_MARGIN + SCENE_KERNEL + 1


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


def crop_origin(top: int) -> int:
    """First row an ffmpeg crop of 4:2:0 video really starts at (odd rows round down)."""
    return top - top % 2


def _frames(
    ffmpeg: str,
    video: Path,
    start: float,
    indices: Sequence[int],
    size: tuple[int, int],
    band: tuple[int, int] | None = None,
) -> Iterator[Image]:
    """Yield the requested decode-order frames (ascending, unique) as BGR arrays.

    With ``band`` (top, height) only those rows are converted and piped, cropped like the band
    audit's decode (identical to cropping the full frame at ``crop_origin(top)``).
    """
    wanted = _selector(indices)
    width, height = size
    crop = "" if band is None else f",crop={width}:{band[1]}:0:{band[0]}"
    command = [ffmpeg, "-nostdin", "-v", "error"]
    command += ["-ss", f"{start:.6f}"] if start else []
    command += ["-i", str(video), "-vf", f"select='{wanted}'{crop}", "-fps_mode", "passthrough"]
    command += ["-frames:v", str(len(indices)), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    height = height if band is None else band[1]
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
    band: tuple[int, int] | None = None,
    cleaned_bands: Sequence[Image] | None = None,
    source_bands: Sequence[Image] | None = None,
) -> list[int]:
    """Unchanged caption-colour pixels per audited frame (``indices`` ascending, unique).

    ``band`` (the band audit's rows) reads only those rows of both videos, and
    ``cleaned_bands`` reuses the band audit's decoded candidate instead of decoding it again;
    ``source_bands`` likewise holds the source's ``band`` rows decoded ahead (``_frames``).
    Both apply only when every audited box lies ``REACH`` rows inside the band, so the counts
    equal the full-frame ones; otherwise the full frames are decoded.
    """
    if list(indices) != sorted(set(indices)) or not indices:
        raise ValueError("Audited frame indices must be ascending and unique")
    if band is not None and not inside(band, [boxes[index] for index in indices]):
        band, cleaned_bands, source_bands = None, None, None
    origin = 0 if band is None else crop_origin(band[0])
    shape = (size[1] if band is None else band[1], size[0])
    originals = (
        iter(source_bands)
        if band is not None and source_bands is not None
        else _frames(ffmpeg, source, source_start, indices, size, band)
    )
    repaired = (
        iter(cleaned_bands)
        if cleaned_bands is not None
        else _frames(ffmpeg, cleaned, 0.0, indices, size, band)
    )
    counts: list[int] = []
    for index, original, clean in zip(indices, originals, repaired, strict=True):
        allowed = box_mask(shape, (0, origin), boxes[index])
        counts.append(unchanged_pink(clean, original, allowed))
    return counts


def inside(band: tuple[int, int], boxes: Sequence[Sequence[Box]]) -> bool:
    """Every box lies ``REACH`` rows inside the rows a crop of ``band`` really holds."""
    first = crop_origin(band[0])
    last = first + band[1]
    return all(first + REACH <= box[1] and box[3] + REACH <= last for row in boxes for box in row)


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
