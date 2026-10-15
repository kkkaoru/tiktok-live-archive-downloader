"""Build a YOLO dataset of avatar faces and source captions from verified evidence.

Frames come from the dated original; caption boxes from the cleanup admissions' genuine OCR
observations (never the bracketing-union interpolation); face boxes from Vision detections and
high-confidence tracker frames. Every frame records which classes it carries, and frames whose face
class is unlabeled are listed rather than treated as negatives silently. The split is by time
blocks with a guard gap.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from replay.core import ReplayError
from replay.yolo_labels import (
    CAPTION,
    CLASS_NAMES,
    FACE,
    PixelBox,
    clip_to_canvas,
    line_box,
    split_by_time_blocks,
    yolo_row,
)

CANVAS = (1080, 1920)
TRACKING_MINIMUM_CONFIDENCE = 0.9
RATE = 30


def caption_boxes(admission_frame: dict[str, object]) -> list[PixelBox]:
    """Line boxes of one genuinely observed subtitle frame, clipped to the canvas."""
    boxes: list[PixelBox] = []
    corners_list = admission_frame.get("lineBoxes")
    if not isinstance(corners_list, list):
        return boxes
    for corners in corners_list:
        clipped = clip_to_canvas(line_box(corners), canvas=CANVAS)
        if clipped is not None:
            boxes.append(clipped)
    return boxes


def face_box(face_frame: dict[str, object]) -> PixelBox | None:
    """The face box of a detection or a high-confidence tracker frame; None otherwise."""
    faces = face_frame.get("faces")
    if not isinstance(faces, list) or len(faces) != 1:
        return None
    method, face = face_frame.get("method"), faces[0]
    if method not in ("detection", "tracking"):
        return None
    if method == "tracking" and face["confidence"] < TRACKING_MINIMUM_CONFIDENCE:
        return None
    region = face["region"]
    return clip_to_canvas(
        PixelBox(region["x"], region["y"], region["width"], region["height"]), canvas=CANVAS
    )


def load_captions(admissions: list[Path]) -> dict[int, list[PixelBox]]:
    """Frame number to caption boxes, for genuine OCR observations only."""
    result: dict[int, list[PixelBox]] = {}
    base = 0
    for path in admissions:
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, frame in enumerate(data["frames"]):
            if frame.get("sampled") and frame.get("classification") == "overlay-subtitle":
                boxes = caption_boxes(frame)
                if boxes:
                    result[base + index] = boxes
        base += len(data["frames"])
    return result


def load_faces(responses: list[Path]) -> dict[int, PixelBox]:
    """Frame number to face box across the ordered raw face windows."""
    result: dict[int, PixelBox] = {}
    base = 0
    for path in responses:
        frames = json.loads(path.read_text(encoding="utf-8"))
        for index, frame in enumerate(frames):
            box = face_box(frame)
            if box is not None:
                result[base + index] = box
        base += len(frames)
    return result


def extract(source: Path, frames: list[int], directory: Path, *, size: int, rate: int) -> None:
    """Write one JPEG per frame number, seeking to each frame instead of decoding the whole file."""
    for number in frames:
        command = [
            "ffmpeg", "-v", "error", "-nostdin", "-n",
            "-ss", f"{number / rate:.6f}", "-i", str(source),
            "-frames:v", "1", "-vf", f"scale={size}:-2", "-q:v", "3",
            str(directory / f"frame-{number:06d}.jpg"),
        ]  # fmt: skip
        subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--admission", type=Path, action="append", required=True)
    parser.add_argument("--face-response", type=Path, action="append", required=True)
    parser.add_argument("--step", type=int, required=True, help="keep one frame per STEP frames")
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--validation-every", type=int, required=True)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit (diagnostics only)")
    args = parser.parse_args(argv)
    captions, faces = load_captions(args.admission), load_faces(args.face_response)
    candidates = [n for n in sorted(captions) if n % args.step == 0]
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        raise ReplayError("No observed subtitle frame matches the step")
    train, validation = split_by_time_blocks(
        len(candidates), block=args.block, validation_every=args.validation_every, gap=args.gap
    )
    args.output.mkdir(mode=0o700, parents=True)
    for part in ("train", "val"):
        (args.output / "images" / part).mkdir(parents=True)
        (args.output / "labels" / part).mkdir(parents=True)
    kept = {"train": train, "val": validation}
    record: list[dict[str, object]] = []
    for part, indices in kept.items():
        numbers = [candidates[i] for i in indices]
        extract(args.source, numbers, args.output / "images" / part, size=args.size, rate=RATE)
        for number in numbers:
            rows = [yolo_row(CAPTION, box, canvas=CANVAS) for box in captions[number]]
            face = faces.get(number)
            if face is not None:
                rows.append(yolo_row(FACE, face, canvas=CANVAS))
            (args.output / "labels" / part / f"frame-{number:06d}.txt").write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )
            record.append(
                {
                    "frame": number,
                    "split": part,
                    "captions": len(captions[number]),
                    "face": face is not None,
                }
            )
    (args.output / "data.yaml").write_text(
        f"path: {args.output}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    (args.output / "frames.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    without_face = sum(1 for r in record if not r["face"])
    print(
        json.dumps(
            {
                "frames": len(record),
                "train": len(train),
                "val": len(validation),
                "framesWithoutFaceLabel": without_face,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
