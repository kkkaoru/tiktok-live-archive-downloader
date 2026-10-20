"""Real collaborators and orchestration of the sampled-OCR repair planning for one source chunk.

``band_triggers`` decodes only the subtitle band and finds abrupt changes, ``session_ocr``
runs the native OCR session tool on the sampled PTS only, ``detector_regions`` runs the
batched detector on every frame, and ``plan_chunk`` joins them with
``subtitle_sampled_admission``. Every stage is timed. Nothing here approves quality.
"""

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from math import floor
from pathlib import Path
from typing import IO

import cv2
import numpy as np

from replay.caption_io import FrameObservation, native_ocr_frames
from replay.caption_masks import MaskPolicy, TextDetection, TextRectangle
from replay.subtitle_cleanup_plan import CaptionSearchProfile
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_detector_mps import MEAN, SCALE, Network, decoder_model
from replay.subtitle_region_fill import Region
from replay.subtitle_sampled_admission import admission
from replay.subtitle_sampled_observations import change_triggers, sample_indices

TOOL = "video_text_recognize_session"
SESSION_FRAMES = 64
OUTPUT_LIMIT = "bounded capture limit"
BAND_MARGIN = 80
BAND_SCALE = (270, 105)
DETECTOR_GRID = (416, 736)
DETECTOR_BATCH = 8
TIMEOUT_SECONDS = 600


class OutputLimitError(ValueError):
    """A single frame alone exceeded the native session's bounded capture limit."""


def _decode(command: list[str]) -> tuple[subprocess.Popen[bytes], IO[bytes]]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if process.stdout is None:
        raise ValueError("Missing decoder output stream")
    return process, process.stdout


def band_triggers(
    source: Path,
    clock: VideoFrameClock,
    *,
    canvas_width: int,
    envelope: tuple[int, int, int, int],
    ffmpeg: str = "ffmpeg",
) -> tuple[int, ...]:
    """Change-trigger frames of the downscaled subtitle band, one decode of the chunk."""
    start = float(clock.pts[0] * clock.time_base)
    top = max(0, envelope[1] - BAND_MARGIN)
    height = envelope[3] + 2 * BAND_MARGIN
    scale = f"scale={BAND_SCALE[0]}:{BAND_SCALE[1]}"
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-ss", f"{max(0.0, start - 0.05):.6f}", "-i",
        str(source), "-vf", f"crop={canvas_width}:{height}:0:{top},{scale}",
        "-frames:v", str(len(clock.pts)), "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]  # fmt: skip
    process, stream = _decode(command)
    data = stream.read()
    if process.wait() != 0:
        raise ValueError("Band decode failed")
    frame = BAND_SCALE[0] * BAND_SCALE[1]
    if len(data) != frame * len(clock.pts):
        raise ValueError("Band decode returned an unexpected frame count")
    bands = np.frombuffer(data, dtype=np.uint8).reshape(-1, BAND_SCALE[1], BAND_SCALE[0])
    return change_triggers(bands)


def session_clock(pts: int, clock: VideoFrameClock) -> dict[str, int]:
    exact = Fraction(pts * clock.time_base)
    return {"timescale": exact.denominator, "value": exact.numerator}


def _session_child(
    native: Path,
    source: Path,
    clock: VideoFrameClock,
    chunk: Sequence[int],
    *,
    canvas: tuple[int, int],
    directory: Path,
    root: Path,
    workers: int,
) -> list[FrameObservation] | None:
    """Run one native session child; ``None`` when it exceeded its bounded capture limit."""
    (directory / "native").mkdir(parents=True, mode=0o700)
    (directory / "scratch").mkdir(mode=0o700)
    request = {
        "checkoutRoot": str(root),
        "path": str(source),
        "allowedSourceRoot": str(source.parent),
        "checkpointDirectory": str(directory / "native"),
        "samples": [session_clock(clock.pts[index], clock) for index in chunk],
        "width": canvas[0],
        "height": canvas[1],
        "maximumSeconds": 60,
        "workers": workers,
    }
    (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
    scratch = str(directory / "scratch")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": scratch,
        "TMP": scratch,
        "TEMP": scratch,
        "XDG_CACHE_HOME": scratch,
    }
    with (
        (directory / "stdout.json").open("xb") as out,
        (directory / "stderr.log").open("xb") as err,
    ):
        code = subprocess.run(
            [str(native), "measure-media", TOOL, str(directory / "request.json")],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env=environment,
            timeout=TIMEOUT_SECONDS,
            check=False,
        ).returncode
    if code != 0:
        if OUTPUT_LIMIT in (directory / "stderr.log").read_text(encoding="utf-8", errors="replace"):
            return None
        raise ValueError("Native OCR session failed; evidence retained")
    blocks = sorted(
        (directory / "native").glob("block-*.json"),
        key=lambda path: int(path.stem.split("-")[1]) if path.stem.count("-") == 1 else -1,
    )
    frames = [
        frame
        for block in blocks
        if block.stem.count("-") == 1
        for frame in native_ocr_frames(json.loads(block.read_text(encoding="utf-8")))
    ]
    if len(frames) != len(chunk):
        raise ValueError("Native OCR session did not cover every sampled frame")
    return frames


def ocr_band(envelope: tuple[int, int, int, int], canvas: tuple[int, int]) -> tuple[int, int]:
    """(top, height) of the caption band, the envelope widened by BAND_MARGIN, inside the canvas."""
    top = max(0, envelope[1] - BAND_MARGIN)
    bottom = min(canvas[1], envelope[1] + envelope[3] + BAND_MARGIN)
    return top, bottom - top


def _band_frame(
    native: Path,
    source: Path,
    clock: VideoFrameClock,
    index: int,
    *,
    canvas: tuple[int, int],
    band: tuple[int, int],
    directory: Path,
    root: Path,
    workers: int,
    ffmpeg: str = "ffmpeg",
) -> FrameObservation:
    """OCR one frame's caption band only, for a frame whose full-frame text exceeds the limit.

    The band is cut from the exact source frame (ProRes 422 HQ, intra-only) into a one-frame
    movie in the child directory (its provenance is recorded next to it), recognised by the
    unchanged native tool and mapped back to the full canvas. Text outside the band is not
    read; nothing here claims the rest of the frame has no text.
    """
    top, height = band
    directory.mkdir(parents=True, mode=0o700)
    seconds = Fraction(clock.pts[index] * clock.time_base)
    start = Fraction(floor(seconds * 1_000_000), 1_000_000)
    movie = directory / "band.mov"
    cut = subprocess.run(
        [
            ffmpeg, "-nostdin", "-v", "error", "-ss", f"{float(start):.6f}", "-i", str(source),
            "-vf", f"crop={canvas[0]}:{height}:0:{top}", "-frames:v", "1",
            "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", str(movie),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )  # fmt: skip
    if cut.returncode != 0 or not movie.is_file():
        raise ValueError("Caption band could not be cut from the source frame")
    (directory / "band.json").write_text(
        json.dumps({"index": index, "sourcePTS": clock.pts[index], "top": top, "height": height}),
        encoding="utf-8",
    )
    frames = _session_child(
        native,
        movie,
        VideoFrameClock((0,), clock.time_base, clock.last_duration),
        [0],
        canvas=(canvas[0], height),
        directory=directory / "ocr",
        root=root,
        workers=workers,
    )
    if frames is None:
        raise ValueError("Caption band exceeded the native capture limit; evidence retained")
    frame = frames[0]
    return replace(
        frame,
        requested_time=float(seconds),
        actual_time=float(seconds),
        width=canvas[0],
        height=canvas[1],
        detections=tuple(
            TextDetection(
                line.text,
                line.confidence,
                TextRectangle(
                    line.rectangle.x,
                    line.rectangle.y + top,
                    line.rectangle.width,
                    line.rectangle.height,
                ),
            )
            for line in frame.detections
        ),
    )


def session_ocr(
    native: Path,
    source: Path,
    clock: VideoFrameClock,
    indices: Sequence[int],
    *,
    canvas: tuple[int, int],
    work: Path,
    root: Path,
    workers: int = 2,
    band: tuple[int, int] | None = None,
) -> dict[int, FrameObservation]:
    """OCR only the given frame indices through the native session tool, 64 per child.

    The native session bounds the text of all its frames together. A child that exceeds that
    bound (dense on-screen text) is not retried unchanged: its frames are halved into new,
    smaller children (the failed child's evidence stays), down to a single frame. A single frame
    that alone exceeds the bound is read through its caption ``band`` (top, height) when given.
    """
    found: dict[int, FrameObservation] = {}
    pending = [
        list(indices[first : first + SESSION_FRAMES])
        for first in range(0, len(indices), SESSION_FRAMES)
    ]
    number = 0
    while pending:
        chunk = pending.pop(0)
        frames = _session_child(
            native,
            source,
            clock,
            chunk,
            canvas=canvas,
            directory=work / f"child-{number:03d}",
            root=root,
            workers=workers,
        )
        number += 1
        if frames is None and len(chunk) == 1:
            if band is None:
                raise OutputLimitError("Native OCR session failed; evidence retained")
            found[chunk[0]] = _band_frame(
                native,
                source,
                clock,
                chunk[0],
                canvas=canvas,
                band=band,
                directory=work / f"band-{number - 1:03d}",
                root=root,
                workers=workers,
            )
            continue
        if frames is None:
            half = len(chunk) // 2
            pending[:0] = [chunk[:half], chunk[half:]]
            continue
        found.update(zip(chunk, frames, strict=True))
    return found


def detector_regions(
    source: Path,
    clock: VideoFrameClock,
    *,
    canvas: tuple[int, int],
    envelope: tuple[int, int, int, int],
    network: Network,
    ffmpeg: str = "ffmpeg",
) -> list[Region | None]:
    """Per-frame detector region: bounding box of the text polygons touching the envelope."""
    width, height = canvas
    start = float(clock.pts[0] * clock.time_base)
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-ss", f"{max(0.0, start - 0.05):.6f}", "-i",
        str(source), "-frames:v", str(len(clock.pts)), "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    decoder = decoder_model()
    decoder.setInputSize(DETECTOR_GRID)
    scale = np.array([width / DETECTOR_GRID[0], height / DETECTOR_GRID[1]])
    process, stream = _decode(command)
    size = width * height * 3
    regions: list[Region | None] = []
    while len(regions) < len(clock.pts):
        group: list[np.ndarray] = []
        while len(group) < DETECTOR_BATCH and len(regions) + len(group) < len(clock.pts):
            data = stream.read(size)
            if len(data) < size:
                raise ValueError("Detector decode ended early")
            group.append(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3))
        blobs = np.stack(
            [
                ((cv2.resize(frame, DETECTOR_GRID).astype(np.float32) - MEAN) * SCALE).transpose(
                    2, 0, 1
                )
                for frame in group
            ]
        )
        probs = network(blobs).reshape(len(group), DETECTOR_GRID[1], DETECTOR_GRID[0])
        for prob in probs:
            polygons, _ = decoder.detect(np.repeat(prob[..., None], 3, axis=2))
            regions.append(_envelope_region(polygons, scale, envelope))
    stream.close()
    if process.wait() != 0:
        raise ValueError("Detector decode failed")
    return regions


def _envelope_region(
    polygons: Sequence[np.ndarray], scale: np.ndarray, envelope: tuple[int, int, int, int]
) -> Region | None:
    top, bottom = envelope[1], envelope[1] + envelope[3]
    boxes = [np.asarray(polygon, dtype=np.float64).reshape(-1, 2) * scale for polygon in polygons]
    touching = [box for box in boxes if box[:, 1].min() < bottom and box[:, 1].max() > top]
    if not touching:
        return None
    stacked = np.concatenate(touching)
    return (
        int(stacked[:, 0].min()),
        int(stacked[:, 1].min()),
        int(np.ceil(stacked[:, 0].max())),
        int(np.ceil(stacked[:, 1].max())),
    )


def plan_chunk(
    *,
    clock: VideoFrameClock,
    profile: CaptionSearchProfile,
    stride: int,
    radius: int,
    triggers: Callable[[], Sequence[int]],
    ocr: Callable[[Sequence[int]], Mapping[int, FrameObservation]],
    regions: Callable[[], Sequence[Region | None]],
) -> dict[str, object]:
    """Join the collaborators into one admission, timing every stage."""
    times: dict[str, float] = {}
    started = time.monotonic()
    found = tuple(triggers())
    times["triggers"] = time.monotonic() - started
    samples = sample_indices(len(clock.pts), stride, found)
    started = time.monotonic()
    sampled = dict(ocr(samples))
    times["ocr"] = time.monotonic() - started
    started = time.monotonic()
    detected = list(regions())
    times["detectorRegions"] = time.monotonic() - started
    result = admission(
        sampled=sampled,
        clock=clock,
        profile=profile,
        policy=MaskPolicy(),
        stride=stride,
        triggers=found,
        detected=detected,
        radius=radius,
    )
    result["stageSeconds"] = {name: round(value, 3) for name, value in times.items()}
    return result
