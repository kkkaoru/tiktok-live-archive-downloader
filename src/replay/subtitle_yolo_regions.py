"""Caption search regions and repair masks of a source chunk from the pinned YOLO detector.

The chunk is decoded once at the detector's training width and every frame is detected in
batches; caption boxes are kept per frame in canvas pixels. They serve two consumers:

- search regions, with the contract of ``subtitle_sampled_chunk.detector_regions``: one region
  per chunk frame, the padded bounding box of the caption boxes touching the measured caption
  envelope, or ``None``;
- repair masks, with the contract of the learned text detector inside the renderer: one 0/255
  mask per active-frame ROI, the padded caption boxes of that frame filled. The renderer visits
  the active frames in order, so the i-th ROI belongs to the i-th planned frame; a count mismatch
  fails instead of masking the wrong frame.

Neither proves removal quality: repair, the OCR audit and the colour-residual gate are unchanged.
"""

import os
import subprocess
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_region_fill import Region
from replay.yolo_joint_dataset import ROW_GAP_LINES, caption_pixels
from replay.yolo_labels import CAPTION
from replay.yolo_track import TRAIN_WIDTH, scaled_height
from replay.yolo_train import check_checkpoint, check_settings

BATCH = 32
CONFIDENCE = 0.25
PAD = 6
MASK_PAD = 4

Image = NDArray[np.uint8]
Corners = tuple[float, float, float, float]
Detector = Callable[[list[Image]], list[list[Corners]]]


@dataclass(frozen=True, slots=True)
class YoloSettings:
    """The pinned detector: checkpoint and digest, Ultralytics settings and inference size."""

    weights: Path
    sha256: str
    config_directory: Path
    work: Path
    imgsz: int = 640
    device: str = "mps"
    masks: bool = False
    admission: bool = False


@dataclass(frozen=True, slots=True)
class PlannedBoxes:
    """One active frame: ROI origin on the canvas and the frame's caption boxes (canvas pixels)."""

    origin: tuple[int, int]
    boxes: tuple[Corners, ...]


def region_of(
    boxes: Sequence[Corners],
    *,
    scale: float,
    envelope: tuple[int, int, int, int],
    canvas: tuple[int, int],
) -> Region | None:
    """Canvas bounding box of the boxes touching the envelope rows, padded and clipped."""
    top, bottom = envelope[1], envelope[1] + envelope[3]
    scaled = [tuple(value * scale for value in box) for box in boxes]
    touching = [box for box in scaled if box[1] < bottom and box[3] > top]
    if not touching:
        return None
    return (
        max(0, int(min(box[0] for box in touching)) - PAD),
        max(0, int(min(box[1] for box in touching)) - PAD),
        min(canvas[0], int(np.ceil(max(box[2] for box in touching))) + PAD),
        min(canvas[1], int(np.ceil(max(box[3] for box in touching))) + PAD),
    )


def chunk_frames(
    source: Path, clock: VideoFrameClock, *, canvas: tuple[int, int], ffmpeg: str
) -> Generator[Image]:
    """Every chunk frame scaled to the training width (BGR), selected exactly as the renderer does.

    Like the repair renderer, frames are chosen by presentation timestamp from the chunk's first
    PTS with timestamps passed through, so the i-th box belongs to the i-th rendered frame even
    across timestamp gaps (a count-based decode after a time seek drifts there).
    """
    height = scaled_height(*canvas)
    seek = max(0, int(clock.pts[0] * clock.time_base) - 1)
    selection = (
        f"settb={clock.time_base},select=gte(pts\\,{clock.pts[0]}),"
        f"trim=end_frame={len(clock.pts)},scale={TRAIN_WIDTH}:{height}"
    )
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-copyts", "-noaccurate_seek", "-ss", str(seek),
        "-i", str(source), "-map", "0:v:0", "-an", "-fps_mode", "passthrough",
        "-vf", selection, "-frames:v", str(len(clock.pts)),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    size = TRAIN_WIDTH * height * 3
    with subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL) as process:
        if process.stdout is None:
            raise ReplayError("ffmpeg gave no output stream")
        while len(block := process.stdout.read(size)) == size:
            yield np.frombuffer(block, dtype=np.uint8).reshape(height, TRAIN_WIDTH, 3)
    if process.returncode != 0:
        raise ReplayError("ffmpeg failed while decoding the chunk")


def canvas_scale(canvas: tuple[int, int]) -> float:
    """Canvas pixels per detector-frame pixel."""
    return canvas[0] / TRAIN_WIDTH


def extend_to_glyphs(image: Image, box: Corners) -> Corners:
    """Grow a caption box sideways along the caption glyphs of its own rows.

    Very long lines reach the frame edges and the detector may box only their middle. Within the
    box's rows, glyph columns (the caption-pixel rule the training labels were built with) that
    continue the line with gaps of at most ``ROW_GAP_LINES`` line heights are added. Rows above
    and below are never added, so a neighbouring line cannot widen this one.
    """
    top, bottom = max(0, int(box[1])), min(image.shape[0], int(np.ceil(box[3])))
    if bottom <= top:
        return box
    columns = caption_pixels(image[top:bottom])[1].any(axis=0)
    gap = max(1, round(ROW_GAP_LINES * (bottom - top)))
    left, right = max(0, int(box[0])), min(image.shape[1] - 1, int(np.ceil(box[2])))
    reach = left
    for column in range(left - 1, -1, -1):
        if reach - column > gap:
            break
        if columns[column]:
            reach = column
    grown_left = reach
    reach = right
    for column in range(right + 1, image.shape[1]):
        if column - reach > gap:
            break
        if columns[column]:
            reach = column
    return (float(min(grown_left, box[0])), box[1], float(max(reach + 1, box[2])), box[3])


def chunk_boxes(
    frames: Iterator[Image], detector: Detector, *, count: int, scale: float
) -> list[list[Corners]]:
    """Caption boxes of every chunk frame in canvas pixels, detecting in batches.

    Each detected box is first grown along its line's glyphs in the detector frame.
    """
    found: list[list[Corners]] = []
    pending: list[Image] = []

    def flush() -> None:
        for image, boxes in zip(pending, detector(pending), strict=True):
            found.append([extend_to_glyphs(image, box) for box in boxes])
        pending.clear()

    for image in frames:
        pending.append(image)
        if len(pending) == BATCH:
            flush()
    if pending:
        flush()
    if len(found) != count:
        raise ReplayError("YOLO boxes do not cover every chunk frame")
    return [
        [(b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale) for b in boxes] for boxes in found
    ]


def yolo_regions(
    boxes: Sequence[Sequence[Corners]],
    *,
    canvas: tuple[int, int],
    envelope: tuple[int, int, int, int],
) -> list[Region | None]:
    """One search region (or None) per chunk frame from its canvas-pixel caption boxes."""
    return [region_of(frame, scale=1.0, envelope=envelope, canvas=canvas) for frame in boxes]


def box_mask(roi: Image, frame: PlannedBoxes) -> Image:
    """0/255 mask over the ROI with the frame's padded caption boxes filled."""
    height, width = roi.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for left, top, right, bottom in frame.boxes:
        x0 = max(0, int(left) - frame.origin[0] - MASK_PAD)
        y0 = max(0, int(top) - frame.origin[1] - MASK_PAD)
        x1 = min(width, int(np.ceil(right)) - frame.origin[0] + MASK_PAD)
        y1 = min(height, int(np.ceil(bottom)) - frame.origin[1] + MASK_PAD)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


class BoxMaskDetector:
    """Repair masks from planned per-frame caption boxes, consumed in render order."""

    def __init__(self, frames: Sequence[PlannedBoxes]) -> None:
        self._frames = frames
        self._next = 0

    @property
    def consumed(self) -> int:
        return self._next

    def detect(self, rois: Sequence[Image]) -> tuple[Image, ...]:
        if self._next + len(rois) > len(self._frames):
            raise ValueError("More ROIs than planned active frames")
        planned = self._frames[self._next : self._next + len(rois)]
        self._next += len(rois)
        return tuple(box_mask(roi, frame) for roi, frame in zip(rois, planned, strict=True))


def caption_boxes(results: Sequence[Any]) -> list[list[Corners]]:
    """Caption-class boxes of Ultralytics results, in input-image pixels."""
    found: list[list[Corners]] = []
    for result in results:
        boxes = result.boxes
        found.append(
            [
                (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
                for label, c in zip(boxes.cls.tolist(), boxes.xyxy.tolist(), strict=True)
                if int(label) == CAPTION
            ]
        )
    return found


def load_detector(settings: YoloSettings) -> Detector:
    """The pinned checkpoint as a batch detector; nothing is downloaded."""
    check_checkpoint(settings.weights, settings.sha256)
    check_settings(settings.config_directory, settings.work)
    os.environ["YOLO_CONFIG_DIR"] = str(settings.config_directory)
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.models.yolo.model import YOLO

    class CpuNmsPredictor(DetectionPredictor):
        """Ultralytics detection with NMS on the CPU (the same boxes).

        On MPS its per-image boolean indexing waits for the GPU once per image; beside a running
        STTN fill that took a 900-frame chunk 9.2 s against 3.4 s with one copy per batch.
        """

        def postprocess(self, preds: Any, img: Any, orig_imgs: Any, **kwargs: Any) -> Any:
            prediction = preds[0] if isinstance(preds, (list, tuple)) else preds
            original: Callable[..., Any] = super().postprocess  # untyped in Ultralytics
            return original(prediction.cpu(), img, orig_imgs, **kwargs)

    model = YOLO(str(settings.weights))

    def detect(images: list[Image]) -> list[list[Corners]]:
        results = model.predict(
            images, imgsz=settings.imgsz, device=settings.device, conf=CONFIDENCE, verbose=False,
            predictor=CpuNmsPredictor,
        )  # fmt: skip
        return caption_boxes(list(results))

    return detect


def settings_from(config: object) -> YoloSettings:
    """Validated detector settings from a chunk configuration's ``regionDetector`` object."""
    if not isinstance(config, dict) or config.get("kind") != "yolo":
        raise ReplayError("regionDetector must be a yolo detector description")
    weights, digest = config.get("weights"), config.get("sha256")
    directory, work = config.get("configDirectory"), config.get("work")
    imgsz, masks = config.get("imgsz", 640), config.get("masks", False)
    admission = config.get("admission", "ocr")
    if admission not in ("ocr", "yolo"):
        raise ReplayError("regionDetector admission must be ocr or yolo")
    if not all(isinstance(value, str) for value in (weights, digest, directory, work)):
        raise ReplayError("regionDetector needs weights, sha256, configDirectory and work")
    if not isinstance(imgsz, int) or not 32 <= imgsz <= 2048 or not isinstance(masks, bool):
        raise ReplayError("regionDetector imgsz or masks is invalid")
    return YoloSettings(
        Path(str(weights)), str(digest), Path(str(directory)), Path(str(work)), imgsz,
        masks=masks,
        admission=admission == "yolo",
    )  # fmt: skip
