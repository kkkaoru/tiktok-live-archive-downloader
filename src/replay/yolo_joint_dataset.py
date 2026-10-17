"""Two-class (avatar_face, source_caption) YOLO dataset over several recordings, one rule for all.

A JSON manifest names, per recording, the dated source video, its ordered Vision face-window
responses, a frame step and a role (``train``, ``val`` or ``test``). Frames are decoded once per
recording at the detector's training width. A frame is kept only when Vision gives it a trusted
face box (detection, or tracking at high confidence), so a visible avatar is never taught as
background. Caption boxes come from the caption pixels themselves (pink fill, or white glyphs
outlined in pink, joined along a text line), not from OCR: OCR missed the pink English lines and
merged lines inconsistently between recordings, which taught real captions as background.
Nothing here is specific to a recording; the manifest is the only per-recording input.
"""

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError
from replay.recordings import record
from replay.yolo_dataset import face_box, read_paths
from replay.yolo_labels import CAPTION, CLASS_NAMES, FACE, PixelBox, clip_to_canvas, yolo_row
from replay.yolo_track import probe_video, video_frames

ROLES = ("train", "val", "test")
FACE_METHODS = ("detection", "tracking")
CAPTION_HUE = (148, 174)
FILL_SATURATION, FILL_VALUE = 90, 170
OUTLINE_SATURATION, OUTLINE_VALUE = 70, 120
WHITE_VALUE, WHITE_SATURATION = 185, 70
OUTLINE_REACH = 7
CAPTION_ZONE_TOP = 0.45
MIN_LINE_HEIGHT = 5
MAX_LINE_HEIGHT_SHARE = 0.05
MIN_LINE_WIDTH = 18
MIN_LINE_ASPECT = 1.8
MIN_GLYPH_DENSITY = 0.08
MIN_FILL_PIXELS = 8
ROW_OVERLAP = 0.6
ROW_GAP_LINES = 1.5
LINE_MARGIN = 2

Image = NDArray[np.uint8]
Corners = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Recording:
    name: str
    source: Path
    face_responses: tuple[Path, ...]
    step: int
    role: str
    face_methods: tuple[str, ...] = FACE_METHODS


def trusted_faces(responses: Sequence[Path], methods: Sequence[str]) -> dict[int, PixelBox]:
    """Frame number to trusted Vision face box, keeping only the named Vision methods.

    Detection boxes are direct observations; tracking boxes are propagated between detections
    and can drift (inflated boxes on some avatars), so an evaluation set may keep detections only.
    """
    result: dict[int, PixelBox] = {}
    base = 0
    for path in responses:
        frames: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(frames, list):
            raise ReplayError("A face response must be a list of frames")
        for index, frame in enumerate(frames):
            box = face_box(frame) if record(frame).get("method") in methods else None
            if box is not None:
                result[base + index] = box
        base += len(frames)
    return result


def caption_pixels(image: Image) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(caption-coloured pixels, caption glyph pixels) of a BGR frame.

    Glyph pixels are the pink outline/fill plus white glyph bodies that touch the pink outline;
    Japanese lines are white with a thin pink edge, English lines are pink.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    in_hue = (hue >= CAPTION_HUE[0]) & (hue <= CAPTION_HUE[1])
    fill = in_hue & (saturation >= FILL_SATURATION) & (value >= FILL_VALUE)
    outline = in_hue & (saturation >= OUTLINE_SATURATION) & (value >= OUTLINE_VALUE)
    white = (value >= WHITE_VALUE) & (saturation <= WHITE_SATURATION)
    kernel = np.ones((OUTLINE_REACH, OUTLINE_REACH), np.uint8)
    near = cv2.dilate(outline.astype(np.uint8), kernel).astype(bool)
    return fill | outline, outline | (white & near)


def merge_row(boxes: Sequence[Corners]) -> list[Corners]:
    """Join boxes of one text line split by a wide gap between words."""
    merged: list[Corners] = []
    for box in sorted(boxes):
        for index, other in enumerate(merged):
            height = min(box[3] - box[1], other[3] - other[1])
            overlap = min(box[3], other[3]) - max(box[1], other[1])
            if overlap >= ROW_OVERLAP * height and box[0] - other[2] <= ROW_GAP_LINES * height:
                merged[index] = (
                    min(other[0], box[0]),
                    min(other[1], box[1]),
                    max(other[2], box[2]),
                    max(other[3], box[3]),
                )
                break
        else:
            merged.append(box)
    return merged


def caption_lines(image: Image) -> list[Corners]:
    """Pixel boxes (left, top, right, bottom) of caption text lines in the lower caption zone."""
    height, width = image.shape[:2]
    coloured, glyphs = caption_pixels(image)
    mask = glyphs.astype(np.uint8)
    mask[: int(height * CAPTION_ZONE_TOP)] = 0
    joiner = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 16), 1))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, joiner), connectivity=8)
    boxes: list[Corners] = []
    for index in range(1, count):
        x, y, w, h = (int(value) for value in stats[index][:4])
        ys, xs = np.nonzero(mask[y : y + h, x : x + w])
        if xs.size == 0:
            continue
        left, top = x + int(xs.min()), y + int(ys.min())
        right, bottom = x + int(xs.max()) + 1, y + int(ys.max()) + 1
        if is_text_line(mask, coloured, (left, top, right, bottom), frame_height=height):
            boxes.append(
                (
                    max(0, left - LINE_MARGIN),
                    max(0, top - LINE_MARGIN),
                    min(width, right + LINE_MARGIN),
                    min(height, bottom + LINE_MARGIN),
                )
            )
    return merge_row(boxes)


def is_text_line(
    mask: NDArray[np.uint8],
    coloured: NDArray[np.bool_],
    box: Corners,
    *,
    frame_height: int,
) -> bool:
    """A glyph extent is a text line when it is short, wide, dense enough and caption-coloured."""
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    if h < MIN_LINE_HEIGHT or h > frame_height * MAX_LINE_HEIGHT_SHARE:
        return False
    if w < max(MIN_LINE_WIDTH, MIN_LINE_ASPECT * h):
        return False
    glyphs = int(mask[top:bottom, left:right].sum())
    fill = int(coloured[top:bottom, left:right].sum())
    return glyphs >= MIN_GLYPH_DENSITY * w * h and fill >= MIN_FILL_PIXELS


def label_rows(image: Image, face: PixelBox, *, canvas: tuple[int, int]) -> list[str]:
    """YOLO rows of one frame: the Vision face box (canvas pixels) and the caption lines."""
    height, width = image.shape[:2]
    scale = width / canvas[0]
    scaled = clip_to_canvas(
        PixelBox(face.x * scale, face.y * scale, face.width * scale, face.height * scale),
        canvas=(width, height),
    )
    if scaled is None:
        raise ReplayError("The face box lies outside the frame")
    rows = [yolo_row(FACE, scaled, canvas=(width, height))]
    for left, top, right, bottom in caption_lines(image):
        rows.append(
            yolo_row(
                CAPTION,
                PixelBox(float(left), float(top), float(right - left), float(bottom - top)),
                canvas=(width, height),
            )
        )
    return rows


def face_responses(item: dict[str, object]) -> tuple[Path, ...]:
    """Ordered Vision window responses: a finalized head-tracking window index or a list file."""
    index, listing = item.get("windowIndex"), item.get("faceResponses")
    if isinstance(index, str) and listing is None:
        windows: object = json.loads(Path(index).read_text(encoding="utf-8")).get("windows")
        if not isinstance(windows, list) or not windows:
            raise ReplayError("The window index lists no windows")
        return tuple(Path(str(record(window)["evidence"])) / "response.json" for window in windows)
    if isinstance(listing, str) and index is None:
        return tuple(read_paths(Path(listing)))
    raise ReplayError("A recording needs exactly one of windowIndex and faceResponses")


def parse_manifest(path: Path) -> tuple[list[Recording], Path]:
    """Recordings and the output directory of a dataset manifest; every field is validated."""
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("recordings"), list):
        raise ReplayError("The manifest needs a recordings list")
    output = data.get("output")
    if not isinstance(output, str):
        raise ReplayError("The manifest needs an output directory")
    recordings: list[Recording] = []
    for item in data["recordings"]:
        if not isinstance(item, dict):
            raise ReplayError("Each recording must be an object")
        name, source, step, role = (item.get(key) for key in ("name", "source", "step", "role"))
        methods = item.get("faceMethods", list(FACE_METHODS))
        if not isinstance(methods, list) or not methods or not set(methods) <= set(FACE_METHODS):
            raise ReplayError("faceMethods must name detection and/or tracking")
        if not isinstance(name, str) or not name.isidentifier():
            raise ReplayError("A recording name must be a plain identifier")
        if not isinstance(source, str):
            raise ReplayError("A recording needs a source")
        if not isinstance(step, int) or step < 1 or role not in ROLES:
            raise ReplayError("A recording needs a positive step and a known role")
        recordings.append(
            Recording(
                name,
                Path(source),
                face_responses(item),
                step,
                str(role),
                tuple(str(method) for method in methods),
            )  # fmt: skip
        )
    if len({item.name for item in recordings}) != len(recordings):
        raise ReplayError("Recording names must be unique")
    return recordings, Path(output)


def write_recording(
    recording: Recording,
    frames: Iterator[tuple[int, Image]],
    faces: dict[int, PixelBox],
    output: Path,
    *,
    canvas: tuple[int, int],
) -> dict[str, int]:
    """Write the kept frames and labels of one recording; return its counts."""
    images, labels = output / "images" / recording.name, output / "labels" / recording.name
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    counts = {"frames": 0, "captionLines": 0, "framesWithCaption": 0}
    for number, image in frames:
        face = faces.get(number)
        if face is None:
            continue
        rows = label_rows(image, face, canvas=canvas)
        stem = f"{recording.name}_{number:07d}"
        if not cv2.imwrite(str(images / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ReplayError("A frame could not be written")
        (labels / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        captions = len(rows) - 1
        counts["frames"] += 1
        counts["captionLines"] += captions
        counts["framesWithCaption"] += captions > 0
    return counts


def data_yaml(output: Path, recordings: Sequence[Recording]) -> str:
    """The Ultralytics description: one image-list file per role."""
    lines = [f"path: {output}"]
    for role in ROLES:
        names = [item.name for item in recordings if item.role == role]
        listing = output / f"{role}.txt"
        files = [
            str(path) for name in names for path in sorted((output / "images" / name).glob("*.jpg"))
        ]
        listing.write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")
        if files:
            lines.append(f"{role}: {listing}")
    lines.append("names:")
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    return "\n".join(lines) + "\n"


def build_recording(
    recording: Recording, output: Path, *, ffmpeg: str, ffprobe: str
) -> dict[str, object]:
    """Decode one recording once and write its labelled frames."""
    width, height, rate = probe_video(recording.source, ffprobe)
    faces = trusted_faces(recording.face_responses, recording.face_methods)
    if not faces:
        raise ReplayError("A recording has no trusted face box")
    frames = video_frames(
        recording.source, width=width, height=height, start=0,
        end=max(faces) + 1, step=recording.step, rate=rate, ffmpeg=ffmpeg,
    )  # fmt: skip
    counts = write_recording(recording, frames, faces, output, canvas=(width, height))
    print(recording.name, json.dumps(counts), flush=True)
    return {"role": recording.role, "step": recording.step, **counts}


def build(
    manifest: Path, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", jobs: int = 1
) -> dict[str, object]:
    """Build the dataset; `jobs` recordings are decoded at the same time."""
    recordings, output = parse_manifest(manifest)
    if output.exists():
        raise ReplayError("Refusing to overwrite the dataset")
    if not recordings or jobs < 1:
        raise ReplayError("The manifest has no recording or jobs is invalid")
    output.mkdir(mode=0o700, parents=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(
            pool.map(
                lambda item: build_recording(item, output, ffmpeg=ffmpeg, ffprobe=ffprobe),
                recordings,
            )
        )
    summary: dict[str, object] = {
        item.name: result for item, result in zip(recordings, results, strict=True)
    }
    (output / "data.yaml").write_text(data_yaml(output, recordings), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--jobs", type=int, default=1, help="recordings decoded concurrently")
    args = parser.parse_args(argv)
    summary = build(args.manifest, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, jobs=args.jobs)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
