"""A copy of a caption-free source with on-screen chat comment boxes filled by STTN.

Applied only when the recording carries the editing intent ``edits/<recording>/cleanup.json``
(``{"version": 1, "remove": "comment-boxes"}``), from the cleanup step of a source the caption
profile found free of a caption band. A comment box is a filled, near-neutral light rectangle
(rounded corners; any hue tint of the scene leaves it grey) of a text line's height holding dark
text strokes; one cut by the frame edge counts too, when at least ``EDGE_BOX`` of its rectangle
is box grey or text dark (an edge-cut highlight or lamp is not). A box panned out sideways
(narrow, a line high, at the left edge) counts as it is; another thin or text-less grey rectangle
at an edge (a box scrolled or panned out) counts only where a full box was seen within
``RECENT`` frames. Each frame's mask is the union of the boxes of its ``NEIGHBOURS`` frames on
either side, grown by ``MARGIN``; frames without a box keep their decoded pixels. Masked pixels
are filled by the pinned STTN model (``replay.subtitle_sttn``, the caption cleanup's fill) in
chunks of ``CHUNK`` frames and bands of rows at most one model tile high, so the fill keeps the
source's scale. A filled region
whose brightness strays more than ``GUARD`` from its surroundings (STTN has nothing to borrow under
a box that stays put) is refilled spatially (Telea).

The whole video is encoded once more (HEVC as ``source_decodable``); every frame keeps its PTS
(set through ``sendcmd``, one line per run of periodic steps) and the audio is copied. The
derived file keeps the source's name in its own directory beside the box evidence and a record
(source and evidence SHA-256, detector and fill settings, code version, output SHA-256); a rerun
reuses it only when the record still matches.
"""

import argparse
import functools
import io
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from replay import gpu_lease, subtitle_sttn, yolo_track
from replay.core import ReplayError
from replay.edit_completion import load_config
from replay.recordings import record
from replay.source_lead_in import BITRATE, COLOUR, sha256, video_pts, video_stream
from replay.yolo_track import probe_video

Box = tuple[int, int, int, int]
Image = NDArray[np.uint8]

DIRECTORY = "source-comment-clean"
INTENT = "cleanup.json"
RECORD = "comment-clean.json"
EVIDENCE = "comment-boxes.json"
CODE_VERSION = "comment-clean-v4"
INTENT_VALUE = {"version": 1, "remove": "comment-boxes"}
# Detector, in shares of the frame so it holds at any size; measured on youtube-6D9cWCW7T0Y.
GREY_SATURATION = 20  # boxes: S <= 20 of 255; the tinted character's hair and skin: S >= 75
GREY_VALUE = 170
DARK_VALUE = 110
CLOSE = (15, 9)
HEIGHTS = (0.03, 0.12)
MIN_WIDTH = 0.05
FILL = 0.8
TEXT = (0.01, 0.5)
EDGE_MIN = 3
# Share of an edge-cut box that is box grey or text dark: comment boxes 0.94-1.0 (both recordings),
# goggle highlights and lamps cut by youtube-FeR-drWHf4c's edge a median of 0.86.
EDGE_BOX = 0.93
RECENT = 15
SCROLL = 0.1
NEIGHBOURS = 2
MARGIN = 6
CHUNK = 450
BAND = subtitle_sttn.TILE_HEIGHT
CONTEXT = 24
MATRIX = "bt709"
# The fill's own growth (subtitle_sttn.MASK_GROWTH), the ring around a filled region compared with
# it, and the brightness gap (of 255) beyond which the region is refilled spatially.
GROWN = np.ones((2 * subtitle_sttn.MASK_GROWTH + 1,) * 2, np.uint8)
RING = np.ones((25, 25), np.uint8)
GUARD = 40
# A partial smear: this share of a region brighter than nearly all (95th percentile) of its ring,
# by BRIGHT_MARGIN; 5% of the 374 filled regions sampled every 2 s of the video are above 0.1.
BRIGHT_SHARE = 0.1
BRIGHT_MARGIN = 15
TELEA_RADIUS = 5
SCALING = "accurate_rnd+full_chroma_int+full_chroma_inp"
# Frames decoded ahead of the scan: enough to hide the decoder's jitter, a few MB each.
AHEAD = 8
# Longest repeating pattern of PTS steps one timing line expresses (a 30 fps clock in whole
# milliseconds steps 33, 33, 34 ms).
PERIODS = 8


def intent(edit_root: Path) -> bool:
    """Whether the recording asks for its comment boxes to be removed (strictly validated)."""
    path = edit_root / INTENT
    if not path.exists():
        return False
    if json.loads(path.read_text(encoding="utf-8")) != INTENT_VALUE:
        raise ReplayError(f"{INTENT} must be {{version: 1, remove: comment-boxes}}")
    return True


def location(edit_root: Path, source: Path) -> Path:
    """The cleaned copy's path: ``<deliveryRoot>/source-comment-clean/<recording>/<name>``."""
    storage = load_config(edit_root).storage_root
    if storage is None:
        raise ReplayError("The recording has no bound storage for a comment-cleaned copy")
    return storage.parent / DIRECTORY / edit_root.name / source.name


def touches(box: Box, width: int, height: int) -> bool:
    return box[0] == 0 or box[1] == 0 or box[2] == width or box[3] == height


def overlaps(box: Box, others: list[Box], reach: int) -> bool:
    return any(
        box[0] < other[2] and other[0] < box[2] and box[1] < other[3] + reach
        and other[1] - reach < box[3]
        for other in others
    )  # fmt: skip


def comment_boxes(frame: Image, recent: list[Box]) -> tuple[list[Box], list[Box]]:
    """(every comment box, the full ones) of a BGR frame; `recent` are earlier full boxes."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grey = ((hsv[..., 1] <= GREY_SATURATION) & (hsv[..., 2] >= GREY_VALUE)).astype(np.uint8)
    closed = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, cv2.getStructuringElement(0, CLOSE))
    count, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    full: list[Box] = []
    found: list[Box] = []
    for x, y, w, h, area in stats[1:count].tolist():
        box = (x, y, x + w, y + h)
        if area < FILL * w * h or min(w, h) < EDGE_MIN:
            continue
        darks = hsv[y : y + h, x : x + w, 2] < DARK_VALUE
        cut = touches(box, width, height)
        if cut and float((darks | (grey[y : y + h, x : x + w] > 0)).mean()) < EDGE_BOX:
            continue
        dark = float(darks.mean())
        if (
            HEIGHTS[0] * height <= h <= HEIGHTS[1] * height
            and w >= MIN_WIDTH * width
            and TEXT[0] <= dark <= TEXT[1]
        ):
            full.append(box)
            found.append(box)
        elif cut and (
            overlaps(box, recent, round(SCROLL * height))
            # A box panned out sideways: too narrow for its text, still a line high.
            or (
                x == 0 and w < MIN_WIDTH * width and HEIGHTS[0] * height <= h <= HEIGHTS[1] * height
            )
        ):
            found.append(box)
    return found, full


def scanner() -> Callable[[Image], list[Box]]:
    """A detector of the comment boxes of each next frame, keeping the recent full boxes."""
    history: list[list[Box]] = []

    def detect(frame: Image) -> list[Box]:
        nonlocal history
        found, full = comment_boxes(frame, [box for seen in history for box in seen])
        history = [*history[1 - RECENT :], full]
        return found

    return detect


def scan(frames: Iterator[Image]) -> list[list[Box]]:
    """Every frame's comment boxes, in order."""
    detect = scanner()
    return [detect(frame) for frame in frames]


def masks(
    boxes: list[list[Box]], start: int, stop: int, shape: tuple[int, int]
) -> NDArray[np.bool_]:
    """Masks of frames [start, stop): their neighbours' boxes, grown by ``MARGIN``."""
    height, width = shape
    result = np.zeros((stop - start, height, width), dtype=bool)
    for index in range(start, stop):
        near = boxes[max(0, index - NEIGHBOURS) : index + NEIGHBOURS + 1]
        for x0, y0, x1, y1 in (box for frame in near for box in frame):
            result[
                index - start, max(0, y0 - MARGIN) : y1 + MARGIN, max(0, x0 - MARGIN) : x1 + MARGIN
            ] = True
    return result


def bands(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Row ranges at most ``BAND`` high covering a chunk's masked rows with ``CONTEXT`` around."""
    rows = np.flatnonzero(mask.any(axis=(0, 2)))
    height = mask.shape[1]
    runs: list[list[int]] = []
    for row in rows.tolist():
        if runs and row <= runs[-1][1] + 2 * CONTEXT:
            runs[-1][1] = row
        else:
            runs.append([row, row])
    ranges: list[tuple[int, int]] = []
    for first, last in runs:
        top, bottom = max(0, first - CONTEXT), min(height, last + 1 + CONTEXT)
        pieces = -(-(bottom - top) // BAND)
        cuts = [top + round(index * (bottom - top) / pieces) for index in range(pieces + 1)]
        ranges += list(zip(cuts[:-1], cuts[1:], strict=True))
    return ranges


def guarded(frame: Image, mask: NDArray[np.bool_]) -> Image:
    """The frame with each filled region far from its surroundings' brightness refilled (Telea).

    A region counts when its mean is ``GUARD`` off its ring's, or when ``BRIGHT_SHARE`` of it is
    lighter than almost all of its ring (a light smear beside dark background and hair).

    STTN has no other frame to borrow from under a box that stays put, and then paints a light
    smear (youtube-6D9cWCW7T0Y at 1170 s: a 94x94 box filled about 150 levels above the navy
    around it); a spatial fill continues the surroundings instead.
    """
    grown = cv2.dilate(mask.astype(np.uint8), GROWN)
    count, labels = cv2.connectedComponents(grown)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    redo = np.zeros_like(grown)
    for label in range(1, count):
        region = labels == label
        ring = cv2.dilate(region.astype(np.uint8), RING).astype(bool) & (grown == 0)
        if not ring.any():
            continue
        lit = float((gray[region] > np.percentile(gray[ring], 95) + BRIGHT_MARGIN).mean())
        if abs(float(gray[region].mean()) - float(gray[ring].mean())) > GUARD or lit > BRIGHT_SHARE:
            redo[region] = 255
    if redo.any():
        frame[redo > 0] = cv2.inpaint(frame, redo, TELEA_RADIUS, cv2.INPAINT_TELEA)[redo > 0]
    return frame


def fill(chunk: Image, mask: NDArray[np.bool_], inpaint: subtitle_sttn.Tiler) -> Image:
    """The chunk with its masked pixels filled, band by band; other pixels are kept."""
    for top, bottom in bands(mask):
        chunk[:, top:bottom] = subtitle_sttn.fill_band(
            inpaint, np.ascontiguousarray(chunk[:, top:bottom]),
            np.ascontiguousarray(mask[:, top:bottom]),
        )  # fmt: skip
    for index in np.flatnonzero(mask.any(axis=(1, 2))).tolist():
        chunk[index] = guarded(chunk[index], mask[index])
    return chunk


def decoder(source: Path, *, ffmpeg: str) -> list[str]:
    return [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-map", "0:v",
        "-fps_mode", "passthrough",
        "-vf", f"scale=in_color_matrix={MATRIX}:flags={SCALING},format=bgr24",
        "-f", "rawvideo", "pipe:1",
    ]  # fmt: skip


def frames_of(command: list[str], shape: tuple[int, int]) -> Iterator[Image]:
    size = shape[0] * shape[1] * 3
    with subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL) as process:
        if not isinstance(process.stdout, io.BufferedReader):
            raise ReplayError("ffmpeg gave no output stream")
        # Each frame is read straight into its own writable array: no bytes object, no copy.
        while process.stdout.readinto(frame := np.empty((*shape, 3), np.uint8)) == size:
            yield frame
    if process.returncode != 0:
        raise ReplayError("ffmpeg failed while decoding frames")


def prefetched(frames: Iterator[Image]) -> Iterator[Image]:
    """`frames` decoded up to ``AHEAD`` ahead on a thread, so the scan overlaps the decode."""
    return (frame for _, frame in yolo_track.prefetched(((0, f) for f in frames), AHEAD))


def periodic_run(pts: list[int], start: int) -> tuple[int, int]:
    """(period, end): the longest run from `start` whose PTS steps repeat with a period of at most
    ``PERIODS`` (the smallest such period); frames [start, end) follow it."""
    best = (0, start + 1)
    for period in range(1, min(PERIODS, len(pts) - start - 1) + 1):
        end = start + period + 1
        while end < len(pts) and (
            pts[end] - pts[end - 1] == pts[end - period] - pts[end - period - 1]
        ):
            end += 1
        if end > best[1]:
            best = (period, end)
    return best


def timing(pts: list[int], rate: Fraction) -> str:
    """``sendcmd`` lines giving raw frame N (shown at N / rate) the source PTS ``pts[N]``.

    One line per run of periodic PTS steps (``setpts`` counts N itself; step q of a period P has
    been taken floor((k + P - 1 - q) / P) times after k frames): ffmpeg parses a script in time
    quadratic in its lines, 86 s for one line per frame of a 6,610 s recording.
    """
    lines: list[str] = []
    start = 0
    while start < len(pts):
        period, end = periodic_run(pts, start)
        terms = "".join(
            f"+{pts[start + q + 1] - pts[start + q]}*floor((N-{start}+{period - 1 - q})/{period})"
            for q in range(period)
        )
        at = max(0.0, float((start - Fraction(1, 2)) / rate))
        lines.append(f"{at:.6f} setpts expr {pts[start]}{terms};\n")
        start = end
    return "".join(lines)


def encoder(
    source: Path, output: Path, script: Path, stream: dict[str, object], shape: tuple[int, int],
    *, rate: Fraction, ffmpeg: str,
) -> list[str]:  # fmt: skip
    if "'" in str(script):
        raise ReplayError("The timing command path must not contain a quote")
    base = Fraction(str(stream["time_base"]))
    colour = [
        argument
        for key, option in COLOUR
        if isinstance(stream.get(key), str) and stream[key] != "unknown"
        for argument in (option, str(stream[key]))
    ]
    return [
        ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{shape[1]}x{shape[0]}", "-framerate", str(rate), "-i", "pipe:0",
        "-i", str(source), "-map", "0:v", "-map", "1:a",
        "-vf", f"sendcmd=f='{script}',settb={base},setpts=0,"
               f"scale=out_color_matrix={MATRIX}:out_range=tv:flags={SCALING},format=yuv420p",
        "-fps_mode", "passthrough", "-enc_time_base", str(base),
        "-video_track_timescale", str(base.denominator), "-c:v", "hevc_videotoolbox",
        "-b:v", BITRATE, "-tag:v", "hvc1", *colour, "-c:a", "copy", "-movflags", "+faststart",
        str(output),
    ]  # fmt: skip


def render(
    source: Path, boxes: list[list[Box]] | None, inpainter: Callable[[], subtitle_sttn.Tiler],
    *, shape: tuple[int, int], encode: list[str], ffmpeg: str, lease: str,
) -> list[list[Box]]:  # fmt: skip
    """Decode once, fill the chunks holding boxes, encode every frame in order; the boxes.

    Without `boxes` each frame is scanned as it is decoded (the same frames ``scan`` reads). A
    chunk is filled under a GPU lease of its own, so the box-less rest of the video holds none;
    `inpainter` is called for each filled chunk (it loads the model once).
    """
    found: list[list[Box]] = [] if boxes is None else boxes
    detect = scanner() if boxes is None else None
    with subprocess.Popen(encode, stdin=subprocess.PIPE) as process:
        sink = process.stdin
        if sink is None:
            raise ReplayError("ffmpeg took no input stream")
        # A chunk and the NEIGHBOURS frames after it whose boxes its masks need, in one buffer
        # filled in place: no list of frames, no stacked copy of it.
        chunk: Image = np.empty((CHUNK + NEIGHBOURS, *shape, 3), np.uint8)
        start = count = 0

        def flush(size: int) -> None:
            nonlocal start, count
            stop = start + size
            frames = chunk[:size]
            if any(found[max(0, start - NEIGHBOURS) : stop + NEIGHBOURS]):
                with gpu_lease.lease(gpu_lease.root(), lease):
                    frames = fill(frames, masks(found, start, stop, shape), inpainter())
            sink.write(frames.data)
            chunk[: count - size] = chunk[size:count]
            start, count = stop, count - size

        for frame in prefetched(frames_of(decoder(source, ffmpeg=ffmpeg), shape)):
            chunk[count] = frame
            if detect is not None:
                found.append(detect(chunk[count]))
            count += 1
            if count == CHUNK + NEIGHBOURS:
                flush(CHUNK)
        while count:
            flush(min(count, CHUNK))
        sink.close()
    if process.returncode != 0 or start != len(found):
        raise ReplayError("The comment-cleaned copy was not encoded completely")
    return found


def settings() -> dict[str, object]:
    return {
        "greySaturation": GREY_SATURATION, "greyValue": GREY_VALUE, "darkValue": DARK_VALUE,
        "close": list(CLOSE), "heights": list(HEIGHTS), "minWidth": MIN_WIDTH, "fill": FILL,
        "text": list(TEXT), "edgeMin": EDGE_MIN, "edgeBox": EDGE_BOX,
        "recent": RECENT, "scroll": SCROLL,
        "neighbours": NEIGHBOURS, "margin": MARGIN, "chunk": CHUNK, "band": BAND,
        "context": CONTEXT, "matrix": MATRIX, "scaling": SCALING, "guard": GUARD,
        "brightShare": BRIGHT_SHARE, "brightMargin": BRIGHT_MARGIN,
    }  # fmt: skip


def cached_boxes(path: Path, digest: str) -> list[list[Box]] | None:
    """The recorded boxes of this source and detector, or None.

    Stale evidence is left in place: only a completed scan replaces it (atomically).
    """
    if not path.exists():
        return None
    found = record(json.loads(path.read_text(encoding="utf-8")))
    if found.get("sourceSHA256") != digest or found.get("detector") != settings():
        return None
    rows = found.get("frames")
    if not isinstance(rows, list):
        raise ReplayError("Malformed comment box evidence")
    return [[(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in row] for row in rows]


def evidence_for(
    source: Path, path: Path, digest: str, shape: tuple[int, int], *, ffmpeg: str
) -> list[list[Box]]:
    """The source's comment boxes per frame, scanned once."""
    boxes = cached_boxes(path, digest)
    if boxes is None:
        boxes = scan(prefetched(frames_of(decoder(source, ffmpeg=ffmpeg), shape)))
        write_evidence(path, digest, boxes)
    return boxes


def write_evidence(path: Path, digest: str, boxes: list[list[Box]]) -> None:
    partial = path.with_name(f"partial-{path.name}")
    partial.write_text(
        json.dumps({"sourceSHA256": digest, "detector": settings(),
                    "frames": [[list(box) for box in row] for row in boxes]}),
        encoding="utf-8",
    )  # fmt: skip
    partial.rename(path)


def prepared(
    source: Path, output: Path, *, sttn: dict[str, object], ffmpeg: str, ffprobe: str,
    lease: str,
) -> Path:  # fmt: skip
    """The verified comment-cleaned copy of `source` at `output`."""
    directory = output.parent
    if directory.resolve() == source.resolve().parent or output.name != source.name:
        raise ReplayError("The comment-cleaned copy needs its own directory and the source's name")
    stream = video_stream(source, ffprobe)
    width, height, rate = probe_video(source, ffprobe)
    shape = (height, width)
    digest = sha256(source)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence, receipt = directory / EVIDENCE, directory / RECORD

    def expected() -> dict[str, object]:
        return {
            "source": str(source),
            "sourceSize": source.stat().st_size,
            "sourceSHA256": digest,
            "evidenceSHA256": sha256(evidence),
            "detector": settings(),
            "sttn": str(sttn["checkpointSHA256"]),
            "codeVersion": CODE_VERSION,
            "output": str(output),
        }

    if receipt.exists():
        # A finished copy is only checked: its evidence is scanned again only when missing.
        if not evidence.exists():
            evidence_for(source, evidence, digest, shape, ffmpeg=ffmpeg)
        recorded = record(json.loads(receipt.read_text(encoding="utf-8")))
        wanted = expected()
        kept = {key: recorded.get(key) for key in wanted}
        if kept != wanted or recorded.get("outputSHA256") != sha256(output):
            raise ReplayError("The comment-cleaned copy no longer matches its record; remove it")
        return output
    pts = video_pts(source, ffprobe)
    boxes = cached_boxes(evidence, digest)
    if boxes is not None and len(pts) != len(boxes):
        raise ReplayError("The comment box evidence does not cover every frame")
    script, partial = directory / "timing-commands.txt", directory / f"partial-{output.name}"
    partial.unlink(missing_ok=True)
    script.write_text(timing(pts, rate), encoding="utf-8")
    encode = encoder(source, partial, script, stream, shape, rate=rate, ffmpeg=ffmpeg)
    inpainter = functools.cache(
        lambda: subtitle_sttn.Inpainter(
            Path(str(sttn["directory"])),
            checkpoint_sha=str(sttn["checkpointSHA256"]),
            architecture_sha=str(sttn["architectureSHA256"]),
        )
    )
    found = render(source, boxes, inpainter, shape=shape, encode=encode, ffmpeg=ffmpeg, lease=lease)
    if boxes is None:
        write_evidence(evidence, digest, found)
    if video_pts(partial, ffprobe) != pts:
        raise ReplayError("The comment-cleaned copy moved a source frame; it is not used")
    partial.rename(output)
    framed = sum(1 for row in found if row)
    receipt.write_text(
        json.dumps({**expected(), "frames": len(found), "framesWithBoxes": framed,
                    "outputSHA256": sha256(output)}, indent=2) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edit_root", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--defaults", type=Path, required=True, help="the cleanup defaults")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    arguments = parser.parse_args(argv)
    if not intent(arguments.edit_root):
        raise ReplayError(f"{arguments.edit_root.name} has no {INTENT}")
    defaults = record(json.loads(arguments.defaults.read_text(encoding="utf-8")))
    output = prepared(
        arguments.source, location(arguments.edit_root, arguments.source),
        sttn=record(defaults["sttn"]), ffmpeg=arguments.ffmpeg, ffprobe=arguments.ffprobe,
        lease=arguments.edit_root.name,
    )  # fmt: skip
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
