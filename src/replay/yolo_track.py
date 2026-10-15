"""Track the avatar face and the burned-in caption boxes over a video with the trained detector.

Frames are decoded once by ffmpeg, scaled to the training width (so the detector sees the same
pixels it was trained on) and passed to the pinned checkpoint in batches. The result is a JSON
evidence file that names the checkpoint and the video by sha256 and holds, per sampled frame, the
single face box chosen by one fixed rule (highest confidence; the number of faces is kept so
ambiguity stays visible) and the caption box count. Boxes are reported in source pixels. The
evidence is detection output, not a measurement: callers decide where it may replace Vision.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError
from replay.yolo_labels import CAPTION, FACE
from replay.yolo_train import check_checkpoint, check_settings

PROTOCOL = "yolo-face-caption-track-v1"
CONFIDENCE_FLOOR = 0.25
TRAIN_WIDTH = 640
MAX_BATCH = 64
FACE_RULE = "highest-confidence face box; faceCount keeps the number of faces above the floor"


@dataclass(frozen=True, slots=True)
class Row:
    frame: int
    face: tuple[float, float, float, float, float] | None
    face_count: int
    captions: int


class ResultLike(Protocol):
    @property
    def boxes(self) -> Any: ...


class ModelLike(Protocol):
    def predict(
        self,
        source: list[NDArray[np.uint8]],
        *,
        imgsz: int,
        device: str,
        conf: float,
        verbose: bool,
    ) -> Sequence[ResultLike]: ...


class Adapter:
    """Hand `track` only results of the expected class, refusing anything else."""

    def __init__(self, predictor: Any, expected: type) -> None:
        self._predictor = predictor
        self._expected = expected

    def predict(
        self,
        source: list[NDArray[np.uint8]],
        *,
        imgsz: int,
        device: str,
        conf: float,
        verbose: bool,
    ) -> Sequence[ResultLike]:
        found = list(
            self._predictor.predict(source, imgsz=imgsz, device=device, conf=conf, verbose=verbose)
        )
        if not found or not all(isinstance(item, self._expected) for item in found):
            raise ReplayError("Unexpected prediction type")
        return found


def row_from(frame: int, boxes: Any, *, scale: float) -> Row:
    """One frame's row from an Ultralytics `Boxes` (or None); boxes are scaled to source pixels."""
    if boxes is None:
        return Row(frame, None, 0, 0)
    labels, scores, corners = boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
    faces = [
        (float(score), [float(value) * scale for value in corner])
        for label, score, corner in zip(labels, scores, corners, strict=True)
        if int(label) == FACE
    ]
    captions = sum(1 for label in labels if int(label) == CAPTION)
    if not faces:
        return Row(frame, None, 0, captions)
    score, corner = max(faces, key=lambda item: item[0])
    return Row(frame, (corner[0], corner[1], corner[2], corner[3], score), len(faces), captions)


def track(
    frames: Iterator[tuple[int, NDArray[np.uint8]]],
    model: ModelLike,
    *,
    scale: float,
    batch: int,
    device: str,
    imgsz: int,
) -> tuple[list[Row], float]:
    """Run the detector over `frames` in batches; return the rows and the seconds spent in it."""
    if not 1 <= batch <= MAX_BATCH:
        raise ReplayError("Invalid batch size")
    rows: list[Row] = []
    seconds = 0.0
    pending: list[tuple[int, NDArray[np.uint8]]] = []

    def flush() -> None:
        nonlocal seconds
        if not pending:
            return
        started = time.monotonic()
        results = model.predict(
            [image for _, image in pending],
            imgsz=imgsz,
            device=device,
            conf=CONFIDENCE_FLOOR,
            verbose=False,
        )
        seconds += time.monotonic() - started
        if len(results) != len(pending):
            raise ReplayError("Detector returned a different number of results")
        rows.extend(
            row_from(number, result.boxes, scale=scale)
            for (number, _), result in zip(pending, results, strict=True)
        )
        pending.clear()

    for item in frames:
        pending.append(item)
        if len(pending) == batch:
            flush()
    flush()
    return rows, seconds


def video_frames(
    source: Path, *, width: int, height: int, start: int, end: int, step: int, ffmpeg: str
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """Decode frames start, start+step, ... below `end`, scaled to the training width (BGR)."""
    if not 0 <= start < end or step < 1:
        raise ReplayError("Invalid frame range or step")
    scaled = scaled_height(width, height)
    select = f"select='gte(n,{start})*lt(n,{end})*not(mod(n-{start},{step}))'"
    command = [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(source),
        "-vf", f"{select},scale={TRAIN_WIDTH}:{scaled}", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]  # fmt: skip
    size = TRAIN_WIDTH * scaled * 3
    with subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL) as process:
        if process.stdout is None:
            raise ReplayError("ffmpeg gave no output stream")
        number = start
        while len(block := process.stdout.read(size)) == size:
            yield number, np.frombuffer(block, dtype=np.uint8).reshape(scaled, TRAIN_WIDTH, 3)
            number += step
    if process.returncode != 0:
        raise ReplayError("ffmpeg failed while decoding frames")


def scaled_height(width: int, height: int) -> int:
    """The even height `scale=TRAIN_WIDTH:-2` gives, as in the training images."""
    if width < 1 or height < 1:
        raise ReplayError("Invalid video size")
    return 2 * round(height * TRAIN_WIDTH / width / 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def document(
    rows: Sequence[Row],
    *,
    weights: tuple[Path, str],
    source: tuple[Path, str],
    size: tuple[int, int],
    rate: int,
    step: int,
    timing: dict[str, float],
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "provenance": "yolo",
        "weights": {"path": str(weights[0]), "sha256": weights[1]},
        "source": {"path": str(source[0]), "sha256": source[1]},
        "width": size[0],
        "height": size[1],
        "rate": rate,
        "step": step,
        "faceRule": FACE_RULE,
        "confidenceFloor": CONFIDENCE_FLOOR,
        "timing": timing,
        "frames": [
            {
                "frame": row.frame,
                "face": list(row.face) if row.face is not None else None,
                "faceCount": row.face_count,
                "captions": row.captions,
            }
            for row in rows
        ],
    }


def probe_video(source: Path, ffprobe: str) -> tuple[int, int, int]:
    """(width, height, integer frame rate) of the first video stream."""
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(source)],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout  # fmt: skip
    stream = json.loads(out)["streams"][0]
    numerator, denominator = (int(part) for part in str(stream["r_frame_rate"]).split("/"))
    if denominator < 1 or numerator % denominator != 0:
        raise ReplayError("Only integer frame rates are supported")
    return int(stream["width"]), int(stream["height"]), numerator // denominator


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--config-directory", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, model: ModelLike) -> dict[str, object]:
    width, height, rate = probe_video(args.source, args.ffprobe)
    started = time.monotonic()
    rows, inference = track(
        video_frames(
            args.source, width=width, height=height, start=args.start, end=args.end,
            step=args.step, ffmpeg=args.ffmpeg,
        ),
        model, scale=width / TRAIN_WIDTH, batch=args.batch, device=args.device, imgsz=args.imgsz,
    )  # fmt: skip
    wall = time.monotonic() - started
    timing = {
        "wallSeconds": wall,
        "inferenceSeconds": inference,
        "decodeAndOtherSeconds": wall - inference,
        "framesPerSecond": len(rows) / wall if wall > 0 else 0.0,
    }
    return document(
        rows,
        weights=(args.weights, args.weights_sha256),
        source=(args.source, sha256(args.source)),
        size=(width, height),
        rate=rate,
        step=args.step,
        timing=timing,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    check_checkpoint(args.weights, args.weights_sha256)
    check_settings(args.config_directory, args.work)
    os.environ["YOLO_CONFIG_DIR"] = str(args.config_directory)
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.model import YOLO

    result = run(args, Adapter(YOLO(str(args.weights)), Results))
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["timing"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
