"""Evaluate detector weights on recordings of a `yolo_joint_dataset` with one fixed set of gates.

Both classes are judged together, per recording, against what the consumers need:

- face: the share of Vision-labelled frames with a detected face and the error of the detected
  face's top edge in canvas pixels (caption placement keeps captions above that edge);
- caption: the share of caption glyph pixels inside labelled lines that the predicted boxes cover
  (an uncovered glyph pixel is a residual after removal), line recall at IoU 0.5, and the share of
  predicted boxes that touch no labelled line (a false box makes removal repaint the picture);
- speed: batched inference milliseconds per frame on the chosen device.

A recording passes only when every gate passes. The thresholds are module constants, identical
for every model and recording.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError
from replay.mps_graph_cache import released_mps_graphs
from replay.yolo_joint_dataset import caption_pixels
from replay.yolo_labels import CAPTION, FACE
from replay.yolo_track import MAX_BATCH
from replay.yolo_train import check_settings

FACE_DETECTION_MINIMUM = 0.95
FACE_TOP_P90_MAXIMUM = 60.0
CAPTION_COVERAGE_MINIMUM = 0.98
CAPTION_FALSE_BOX_MAXIMUM = 0.05
MILLISECONDS_MAXIMUM = 60.0
MATCH_IOU = 0.5
CONFIDENCE = 0.25

Box = tuple[float, float, float, float]
Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Prediction:
    faces: tuple[tuple[Box, float], ...]
    captions: tuple[Box, ...]


@dataclass(slots=True)
class Tally:
    face_frames: int = 0
    face_found: int = 0
    glyphs: int = 0
    glyphs_covered: int = 0
    lines: int = 0
    lines_found: int = 0
    boxes: int = 0
    false_boxes: int = 0


class Detector(Protocol):
    def __call__(self, images: list[Image]) -> list[Prediction]: ...


def iou(first: Box, second: Box) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    inter = width * height
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


def touches(box: Box, others: Sequence[Box]) -> bool:
    return any(iou(box, other) > 0 for other in others)


def read_labels(path: Path, *, width: int, height: int) -> tuple[list[Box], list[Box]]:
    """(face boxes, caption boxes) of a YOLO label file in image pixels."""
    faces: list[Box] = []
    captions: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label, cx, cy, w, h = line.split()
        x, y, bw, bh = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        box = (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)
        (faces if int(label) == FACE else captions).append(box)
    return faces, captions


def box_mask(boxes: Sequence[Box], shape: tuple[int, int]) -> NDArray[np.bool_]:
    mask = np.zeros(shape, dtype=bool)
    for left, top, right, bottom in boxes:
        rows = slice(max(0, int(top)), int(np.ceil(bottom)))
        mask[rows, max(0, int(left)) : int(np.ceil(right))] = True
    return mask


def read_image(path: Path) -> Image:
    image = cv2.imread(str(path))
    if image is None:
        raise ReplayError("An evaluation image could not be read")
    return np.asarray(image, dtype=np.uint8)


def score_frame(
    tally: Tally,
    image: Image,
    labels: tuple[list[Box], list[Box]],
    prediction: Prediction,
    top_errors: list[float],
    *,
    scale: float,
) -> None:
    """Add one frame's outcome to the recording's tally."""
    faces, captions = labels
    if faces:
        tally.face_frames += 1
        if prediction.faces:
            tally.face_found += 1
            best = max(prediction.faces, key=lambda item: item[1])[0]
            top_errors.append(abs(best[1] - faces[0][1]) * scale)
    shape = (image.shape[0], image.shape[1])
    glyphs = caption_pixels(image)[1] & box_mask(captions, shape)
    tally.glyphs += int(glyphs.sum())
    tally.glyphs_covered += int((glyphs & box_mask(prediction.captions, shape)).sum())
    tally.lines += len(captions)
    tally.lines_found += sum(
        any(iou(line, box) >= MATCH_IOU for box in prediction.captions) for line in captions
    )
    tally.boxes += len(prediction.captions)
    tally.false_boxes += sum(not touches(box, captions) for box in prediction.captions)


def percentile(values: Sequence[float], share: float) -> float | None:
    ordered = sorted(values)
    return round(ordered[int(share * (len(ordered) - 1))], 1) if ordered else None


def report(tally: Tally, top_errors: list[float], milliseconds: float) -> dict[str, object]:
    """The recording's metrics and its gate outcomes."""
    detection = tally.face_found / max(1, tally.face_frames)
    p90 = percentile(top_errors, 0.9)
    coverage = tally.glyphs_covered / max(1, tally.glyphs)
    false_share = tally.false_boxes / max(1, tally.boxes)
    gates = {
        "face": detection >= FACE_DETECTION_MINIMUM
        and p90 is not None
        and p90 <= FACE_TOP_P90_MAXIMUM,
        "caption": coverage >= CAPTION_COVERAGE_MINIMUM
        and false_share <= CAPTION_FALSE_BOX_MAXIMUM,
        "speed": milliseconds <= MILLISECONDS_MAXIMUM,
    }
    return {
        "face": {
            "frames": tally.face_frames,
            "detectionRate": round(detection, 4),
            "topErrorCanvasPx": {"p50": percentile(top_errors, 0.5), "p90": p90},
        },
        "caption": {
            "glyphCoverage": round(coverage, 4),
            "lines": tally.lines,
            "lineRecallIoU50": round(tally.lines_found / max(1, tally.lines), 4),
            "predictedBoxes": tally.boxes,
            "falseBoxShare": round(false_share, 4),
        },
        "msPerFrame": round(milliseconds, 2),
        "pass": {**gates, "joint": all(gates.values())},
    }


def evaluate(
    detector: Detector,
    files: Sequence[Path],
    labels: Path,
    *,
    canvas_width: int,
    batch: int,
) -> dict[str, object]:
    """Run the detector over the images of one recording in batches and score them."""
    if not files or not 1 <= batch <= MAX_BATCH:
        raise ReplayError("No images to evaluate or invalid batch size")
    tally, seconds = Tally(), 0.0
    top_errors: list[float] = []
    for first in range(0, len(files), batch):
        group = files[first : first + batch]
        images = [read_image(path) for path in group]
        started = time.perf_counter()
        predictions = detector(images)
        seconds += time.perf_counter() - started
        for path, image, prediction in zip(group, images, predictions, strict=True):
            height, width = image.shape[:2]
            truth = read_labels(labels / f"{path.stem}.txt", width=width, height=height)
            score_frame(tally, image, truth, prediction, top_errors, scale=canvas_width / width)
    return {"frames": len(files), **report(tally, top_errors, 1000 * seconds / len(files))}


def predictions_from(results: Sequence[Any]) -> list[Prediction]:
    """Ultralytics results as plain boxes per class."""
    found: list[Prediction] = []
    for result in results:
        boxes = result.boxes
        labels, scores, corners = boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        faces = tuple(
            ((float(c[0]), float(c[1]), float(c[2]), float(c[3])), float(score))
            for label, score, c in zip(labels, scores, corners, strict=True)
            if int(label) == FACE
        )
        captions = tuple(
            (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            for label, c in zip(labels, corners, strict=True)
            if int(label) == CAPTION
        )
        found.append(Prediction(faces, captions))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recording", action="append", required=True)
    parser.add_argument("--config-directory", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--canvas-width", type=int, default=1080)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ReplayError("Refusing to overwrite the output")
    check_settings(args.config_directory, args.work)
    os.environ["YOLO_CONFIG_DIR"] = str(args.config_directory)
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    from ultralytics.models.yolo.model import YOLO

    with released_mps_graphs():
        model = YOLO(str(args.weights))

        def detect(images: list[Image]) -> list[Prediction]:
            results = model.predict(
                images, imgsz=args.imgsz, device=args.device, conf=CONFIDENCE, verbose=False
            )
            return predictions_from(list(results))

        detect([read_image(next((args.dataset / "images").rglob("*.jpg")))])  # warm-up
        result = {
            name: evaluate(
                detect,
                sorted((args.dataset / "images" / name).glob("*.jpg")),
                args.dataset / "labels" / name,
                canvas_width=args.canvas_width,
                batch=args.batch,
            )
            for name in args.recording
        }
    document = {"weights": str(args.weights), "imgsz": args.imgsz, "recordings": result}
    args.output.write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(json.dumps(document, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
