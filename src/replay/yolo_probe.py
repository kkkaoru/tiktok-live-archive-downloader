"""Run a trained avatar-face / source-caption detector on frames Vision could not label.

Reports how often each class is found, the per-frame inference time and writes the frames with the
detected boxes drawn on them so a person can judge them: frames without a face label have no
ground truth, so only visual review can say whether a detection is right. The model is loaded from
a pinned checkpoint and the Ultralytics import happens only after the paths are checked.
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

from replay.core import ReplayError
from replay.mps_graph_cache import released_mps_graphs
from replay.yolo_labels import CAPTION, FACE

CONFIDENCE_FLOOR = 0.25


@dataclass(frozen=True, slots=True)
class Detection:
    label: int
    confidence: float
    box: tuple[float, float, float, float]


def summarize(per_frame: Sequence[Sequence[Detection]]) -> dict[str, object]:
    """Share of frames with at least one face / caption box and the best face confidence."""
    if not per_frame:
        raise ReplayError("No frames to summarize")
    total = len(per_frame)
    with_face = [d for frame in per_frame if any(x.label == FACE for x in frame) for d in [frame]]
    with_caption = sum(1 for frame in per_frame if any(x.label == CAPTION for x in frame))
    faces = [x.confidence for frame in per_frame for x in frame if x.label == FACE]
    return {
        "frames": total,
        "framesWithFace": len(with_face),
        "faceShare": len(with_face) / total,
        "framesWithCaption": with_caption,
        "captionShare": with_caption / total,
        "bestFaceConfidencePerFrameMean": (
            sum(max(x.confidence for x in frame if x.label == FACE) for frame in with_face)
            / len(with_face)
            if with_face
            else None
        ),
        "faceBoxCount": len(faces),
    }


class BoxLike(Protocol):
    """The three tensors of Ultralytics' Boxes that the probe reads."""

    @property
    def cls(self) -> Any: ...

    @property
    def conf(self) -> Any: ...

    @property
    def xyxy(self) -> Any: ...


class ResultLike(Protocol):
    """One frame's prediction: its boxes and a way to save the annotated image."""

    @property
    def boxes(self) -> BoxLike | None: ...

    def save(self, *, filename: str) -> str: ...


class ModelLike(Protocol):
    def predict(
        self, source: str, *, imgsz: int, device: str, conf: float, verbose: bool
    ) -> Sequence[ResultLike]: ...


def run_frames(
    model: ModelLike,
    images: Sequence[Path],
    output: Path,
    *,
    imgsz: int,
    device: str,
    draw: int,
) -> tuple[list[list[Detection]], list[float]]:
    """Predict each image; return the detections and the seconds each prediction took."""
    per_frame: list[list[Detection]] = []
    seconds: list[float] = []
    for index, image in enumerate(images):
        started = time.monotonic()
        results = model.predict(
            str(image), imgsz=imgsz, device=device, conf=CONFIDENCE_FLOOR, verbose=False
        )
        seconds.append(time.monotonic() - started)
        per_frame.append(detections_from(results[0].boxes))
        if index < draw:
            results[0].save(filename=str(output / image.name))
    return per_frame, seconds


def detections_from(boxes: BoxLike | None) -> list[Detection]:
    """Detections of one frame; a frame without any box has none."""
    if boxes is None:
        return []
    return [
        Detection(int(c), float(p), (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        for c, p, b in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
        )
    ]


def inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def parse_and_check(argv: list[str] | None) -> tuple[argparse.Namespace, list[dict[str, object]]]:
    """Parse the arguments, check every path and choose the frames to run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("frames", type=Path, help="frames.json of the dataset")
    parser.add_argument("images", type=Path, help="directory holding the dataset images")
    parser.add_argument("output", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config-directory", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--faceless-only", action="store_true")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--draw", type=int, default=12)
    args = parser.parse_args(argv)
    for path in (args.weights, args.frames, args.images, args.output):
        if not inside(path, args.work):
            raise ReplayError("Every path must stay under the work directory")
    if args.output.exists():
        raise ReplayError("Output directory already exists")
    record = json.loads(args.frames.read_text(encoding="utf-8"))
    chosen = [
        item
        for item in record
        if item["split"] == args.split and (not args.faceless_only or not item["face"])
    ]
    if not chosen:
        raise ReplayError("No frame matches the requested split")
    return args, chosen


def report(
    model: ModelLike, args: argparse.Namespace, chosen: Sequence[dict[str, object]]
) -> dict[str, object]:
    """Run the chosen frames, write the annotated images and summary.json, return the summary."""
    args.output.mkdir(mode=0o700, parents=True)
    images = [args.images / args.split / f"frame-{item['frame']:06d}.jpg" for item in chosen]
    per_frame, seconds = run_frames(
        model, images, args.output, imgsz=args.imgsz, device=args.device, draw=args.draw
    )
    warm = seconds[1:] or seconds
    summary = summarize(per_frame) | {
        "split": args.split,
        "facelessOnly": args.faceless_only,
        "msPerFrameMedianWarm": 1000 * sorted(warm)[len(warm) // 2],
        "device": args.device,
        "imgsz": args.imgsz,
        "confidenceFloor": CONFIDENCE_FLOOR,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


class Adapter:
    """Hand `run_frames` only results of the expected class, refusing anything else."""

    def __init__(self, predictor: Any, expected: type) -> None:
        self._predictor = predictor
        self._expected = expected

    def predict(
        self, source: str, *, imgsz: int, device: str, conf: float, verbose: bool
    ) -> Sequence[ResultLike]:
        found = list(
            self._predictor.predict(source, imgsz=imgsz, device=device, conf=conf, verbose=verbose)
        )
        if not found or not all(isinstance(item, self._expected) for item in found):
            raise ReplayError("Unexpected prediction type")
        return found


def main(argv: list[str] | None = None) -> int:
    args, chosen = parse_and_check(argv)
    os.environ["YOLO_CONFIG_DIR"] = str(args.config_directory)
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.model import YOLO

    with released_mps_graphs():
        model = Adapter(YOLO(str(args.weights)), Results)
        print(json.dumps(report(model, args, chosen)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
