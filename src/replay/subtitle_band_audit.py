"""OCR audit of a repaired chunk on the caption band only, several audit frames per OCR image.

The native OCR session's cost is per image, not per pixel, so the caption band of ``tile``
consecutive audit frames is stacked into one image and recognised once. Each recognised line is
mapped back to its own audit frame and to canvas coordinates, so the unchanged residual rule
(confident lines touching the frame's planned region) judges the result. The band is the measured
caption envelope grown by half its height and by the OCR band margin on each side; every audit
region lies inside it. On a deliberately unrepaired chunk the band audit flagged frames that the
full-frame audit read only with low confidence. A stacked image whose lines alone exceed the
native capture bound is not retried unchanged: the chunk is read again untiled, in its own
``untiled`` directory, so each image holds one band. Nothing here approves removal quality.
"""

import json
import subprocess
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.caption_io import FrameObservation
from replay.caption_masks import TextDetection, TextRectangle
from replay.core import ReplayError
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_sampled_chunk import OutputLimitError

MAX_TILE = 4
BAND_MARGIN = 80
MOVIE_RATE = 30

Image = NDArray[np.uint8]
Recognise = Callable[[Path, VideoFrameClock, int], dict[int, FrameObservation]]


def audit_band(envelope: tuple[int, int, int, int], canvas: tuple[int, int]) -> tuple[int, int]:
    """(top, even height) of the band holding the caption zone grown by half its height."""
    grow = envelope[3] // 2 + BAND_MARGIN
    top = max(0, envelope[1] - grow)
    bottom = min(canvas[1], envelope[1] + envelope[3] + grow)
    height = bottom - top
    return top, height - height % 2


def band_frames(
    video: Path,
    indices: Sequence[int],
    *,
    canvas: tuple[int, int],
    band: tuple[int, int],
    ffmpeg: str,
) -> list[Image]:
    """The band of each audited frame of the repaired chunk, in index order, from one decode."""
    if not indices or list(indices) != sorted(set(indices)):
        raise ReplayError("Audit indices must be ascending and unique")
    top, height = band
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0",
        "-vf", f"crop={canvas[0]}:{height}:0:{top}", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    size = canvas[0] * height * 3
    wanted, last = set(indices), indices[-1]
    kept: list[Image] = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL) as process:
        if process.stdout is None:
            raise ReplayError("ffmpeg gave no output stream")
        number = 0
        while number <= last and len(block := process.stdout.read(size)) == size:
            if number in wanted:
                kept.append(np.frombuffer(block, dtype=np.uint8).reshape(height, canvas[0], 3))
            number += 1
        process.stdout.close()
        process.wait()
    if len(kept) != len(indices):
        raise ReplayError("The repaired chunk did not yield every audited band")
    return kept


def write_tiles(bands: Sequence[Image], tile: int, movie: Path, *, ffmpeg: str) -> int:
    """Stack ``tile`` bands per image (the last image padded black) into a new ProRes movie."""
    if not 1 <= tile <= MAX_TILE or not bands:
        raise ReplayError("Invalid audit tile or no bands")
    height, width = bands[0].shape[:2]
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height * tile}", "-r", str(MOVIE_RATE), "-i", "-",
        "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", str(movie),
    ]  # fmt: skip
    images = 0
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        if encoder.stdin is None:
            raise ReplayError("The tile encoder has no input stream")
        for first in range(0, len(bands), tile):
            group = list(bands[first : first + tile])
            group += [np.zeros_like(bands[0])] * (tile - len(group))
            encoder.stdin.write(np.vstack(group).tobytes())
            images += 1
        encoder.stdin.close()
    if encoder.returncode != 0:
        raise ReplayError("The tile encoder failed")
    return images


def movie_clock(movie: Path, *, ffprobe: str) -> VideoFrameClock:
    """Presentation clock of the tile movie (one PTS per stacked image)."""
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=time_base:packet=pts", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout  # fmt: skip
    data = json.loads(out)
    pts = tuple(sorted(int(packet["pts"]) for packet in data["packets"]))
    if len(pts) < 1:
        raise ReplayError("The tile movie has no frames")
    base = Fraction(str(data["streams"][0]["time_base"]))
    step = pts[1] - pts[0] if len(pts) > 1 else int(1 / (MOVIE_RATE * base))
    return VideoFrameClock(pts, base, step)


def untile(
    observed: dict[int, FrameObservation],
    indices: Sequence[int],
    *,
    tile: int,
    band: tuple[int, int],
    canvas: tuple[int, int],
    clock: VideoFrameClock,
) -> dict[int, FrameObservation]:
    """Per audited frame, its own lines in canvas coordinates; every audited frame is present."""
    top, height = band
    lines: dict[int, list[TextDetection]] = {index: [] for index in indices}
    for image, frame in observed.items():
        for line in frame.detections:
            box = line.rectangle
            slot = min(tile - 1, max(0, int((box.y + box.height / 2) // height)))
            position = image * tile + slot
            if position >= len(indices):
                continue
            local = max(0.0, box.y - slot * height)
            lines[indices[position]].append(
                TextDetection(
                    line.text,
                    line.confidence,
                    TextRectangle(box.x, local + top, box.width, min(box.height, height - local)),
                )
            )
    return {
        index: FrameObservation(
            float((clock.pts[index] - clock.pts[0]) * clock.time_base),
            float((clock.pts[index] - clock.pts[0]) * clock.time_base),
            canvas[0],
            canvas[1],
            tuple(lines[index]),
        )
        for index in indices
    }


def band_audit(
    video: Path,
    indices: Sequence[int],
    *,
    clock: VideoFrameClock,
    canvas: tuple[int, int],
    envelope: tuple[int, int, int, int],
    tile: int,
    work: Path,
    recognise: Recognise,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    decoded: list[Image] | None = None,
) -> dict[int, FrameObservation]:
    """Audit observations of the repaired chunk, read from tiled caption bands.

    ``decoded`` receives the decoded bands, so a later check can reuse them.
    """
    band = audit_band(envelope, canvas)
    work.mkdir(parents=True, mode=0o700)
    bands = band_frames(video, indices, canvas=canvas, band=band, ffmpeg=ffmpeg)
    if decoded is not None:
        decoded[:] = bands
    movie = work / "band-tiles.mov"
    images = write_tiles(bands, tile, movie, ffmpeg=ffmpeg)
    tiles_clock = movie_clock(movie, ffprobe=ffprobe)
    if len(tiles_clock.pts) != images:
        raise ReplayError("The tile movie lost an image")
    (work / "band.json").write_text(
        json.dumps({"top": band[0], "height": band[1], "tile": tile, "indices": list(indices)}),
        encoding="utf-8",
    )
    try:
        observed = recognise(movie, tiles_clock, band[1] * tile)
    except OutputLimitError:
        if tile == 1:
            raise
        return band_audit(
            video, indices, clock=clock, canvas=canvas, envelope=envelope, tile=1,
            work=work / "untiled", recognise=recognise, ffmpeg=ffmpeg, ffprobe=ffprobe,
        )  # fmt: skip
    return untile(observed, indices, tile=tile, band=band, canvas=canvas, clock=clock)
