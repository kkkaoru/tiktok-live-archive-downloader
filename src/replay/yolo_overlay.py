"""Render a review video of the pinned detector's boxes over a segment of a recording.

The segment is decoded once at the detector's training width; every frame is detected (batched)
and drawn: the avatar face in green with its top edge marked (the value caption placement uses),
source-caption lines in yellow, each with its confidence, plus the frame time and the measured
detection speed. The video is H.264 at the source frame rate, written to a new file only. It is a
visual check for people, not evidence that any stage passed.
"""

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError
from replay.mps_graph_cache import released_mps_graphs
from replay.yolo_joint_eval import Prediction, predictions_from
from replay.yolo_track import MAX_BATCH, probe_video, video_frames
from replay.yolo_train import check_checkpoint, check_settings

FACE_COLOUR = (60, 220, 60)
CAPTION_COLOUR = (0, 220, 255)
TEXT_COLOUR = (255, 255, 255)
CONFIDENCE = 0.25
FONT = cv2.FONT_HERSHEY_SIMPLEX

Image = NDArray[np.uint8]
Detect = Callable[[list[Image]], list[Any]]


def draw(
    image: Image, prediction: Prediction, *, caption_scores: Sequence[float], header: str
) -> Image:
    """A copy of the frame with the boxes, the face top line and a header drawn on it."""
    canvas = image.copy()
    for (left, top, right, bottom), score in prediction.faces:
        cv2.rectangle(canvas, (int(left), int(top)), (int(right), int(bottom)), FACE_COLOUR, 2)
        cv2.line(canvas, (0, int(top)), (canvas.shape[1], int(top)), FACE_COLOUR, 1)
        cv2.putText(canvas, f"face {score:.2f}", (int(left), max(12, int(top) - 6)), FONT, 0.45,
                    FACE_COLOUR, 1, cv2.LINE_AA)  # fmt: skip
    for (left, top, right, bottom), score in zip(prediction.captions, caption_scores, strict=True):
        cv2.rectangle(canvas, (int(left), int(top)), (int(right), int(bottom)), CAPTION_COLOUR, 1)
        cv2.putText(canvas, f"{score:.2f}", (int(right) + 2, int(bottom)), FONT, 0.35,
                    CAPTION_COLOUR, 1, cv2.LINE_AA)  # fmt: skip
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(canvas, header, (6, 16), FONT, 0.45, TEXT_COLOUR, 1, cv2.LINE_AA)
    return canvas


def caption_scores(results: Sequence[Any]) -> list[list[float]]:
    """Caption-class confidences per result, in the order `predictions_from` keeps the boxes."""
    scores: list[list[float]] = []
    for result in results:
        boxes = result.boxes
        scores.append(
            [
                float(score)
                for label, score in zip(boxes.cls.tolist(), boxes.conf.tolist(), strict=True)
                if int(label) == 1
            ]
        )
    return scores


def encoder(
    output: Path, *, size: tuple[int, int], rate: int, ffmpeg: str
) -> subprocess.Popen[bytes]:
    command = [
        ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{size[0]}x{size[1]}", "-r", str(rate), "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]  # fmt: skip
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def render(
    frames: Iterator[tuple[int, Image]],
    detect: Detect,
    sink: subprocess.Popen[bytes],
    *,
    batch: int,
    label: str,
) -> dict[str, float]:
    """Detect and draw every frame, streaming to the encoder; return timing."""
    if not 1 <= batch <= MAX_BATCH:
        raise ReplayError("Invalid batch size")
    stream = sink.stdin
    if stream is None:
        raise ReplayError("The encoder has no input stream")
    count, seconds = 0, 0.0
    pending: list[tuple[int, Image]] = []

    def flush() -> None:
        nonlocal count, seconds
        started = time.perf_counter()
        results = detect([image for _, image in pending])
        seconds += time.perf_counter() - started
        predictions, scores = predictions_from(results), caption_scores(results)
        speed = 1000 * seconds / (count + len(pending))
        for (number, image), prediction, score in zip(pending, predictions, scores, strict=True):
            header = f"{label}  frame {number:7d}  YOLO {speed:5.1f} ms/frame"
            stream.write(draw(image, prediction, caption_scores=score, header=header).tobytes())
        count += len(pending)
        pending.clear()

    for item in frames:
        pending.append(item)
        if len(pending) == batch:
            flush()
    if pending:
        flush()
    stream.close()
    if sink.wait() != 0:
        raise ReplayError("The encoder failed")
    return {"frames": float(count), "detectionMsPerFrame": 1000 * seconds / max(1, count)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--config-directory", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    if args.start_seconds < 0 or args.seconds <= 0:
        raise ReplayError("Invalid segment")
    check_checkpoint(args.weights, args.weights_sha256)
    check_settings(args.config_directory, args.work)
    width, height, rate = probe_video(args.source, args.ffprobe)
    start = round(args.start_seconds * rate)
    frames = video_frames(
        args.source, width=width, height=height, start=start,
        end=start + round(args.seconds * rate), step=1, rate=rate, ffmpeg=args.ffmpeg,
    )  # fmt: skip
    first = next(frames)
    os.environ["YOLO_CONFIG_DIR"] = str(args.config_directory)
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    from ultralytics.models.yolo.model import YOLO

    with released_mps_graphs():
        model = YOLO(str(args.weights))

        def detect(images: list[Image]) -> list[Any]:
            found = model.predict(
                images, imgsz=args.imgsz, device=args.device, conf=CONFIDENCE, verbose=False
            )
            return list(found)

        detect([first[1]])  # warm-up, not timed
        size = (first[1].shape[1], first[1].shape[0])
        sink = encoder(args.output, size=size, rate=rate, ffmpeg=args.ffmpeg)

        def chained() -> Iterator[tuple[int, Image]]:
            yield first
            yield from frames

        timing = render(chained(), detect, sink, batch=args.batch, label=args.label)
    print(timing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
