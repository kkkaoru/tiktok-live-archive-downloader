"""Build a YOLO dataset of avatar faces and source captions from verified evidence.

Frames come from the dated original; caption boxes from the cleanup admissions' genuine OCR
observations (never the bracketing-union interpolation); face boxes from Vision detections and
high-confidence tracker frames. Every frame records which classes it carries, and frames whose face
class is unlabeled are listed rather than treated as negatives silently. The split is by time
blocks with a guard gap.
"""

import argparse
import json
import os
import shutil
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


def read_paths(path: Path | None) -> list[Path]:
    """Ordered paths from a list file (one per line, blank lines ignored)."""
    if path is None:
        return []
    return [Path(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_faced(source: Path, output: Path) -> int:
    """Copy only frames that carry a trusted face label into a new two-class dataset.

    A frame without a face label shows the avatar often enough that training on it would teach the
    detector that a visible face is background, so such frames are left out of this dataset.
    """
    record = json.loads((source / "frames.json").read_text(encoding="utf-8"))
    kept = [item for item in record if item["face"]]
    if not kept:
        raise ReplayError("No frame carries a trusted face label")
    for part in ("train", "val"):
        (output / "images" / part).mkdir(parents=True)
        (output / "labels" / part).mkdir(parents=True)
    for item in kept:
        name, part = f"frame-{item['frame']:06d}", item["split"]
        shutil.copy2(
            source / "images" / part / f"{name}.jpg", output / "images" / part / f"{name}.jpg"
        )
        shutil.copy2(
            source / "labels" / part / f"{name}.txt", output / "labels" / part / f"{name}.txt"
        )
    data = (source / "data.yaml").read_text(encoding="utf-8").replace(str(source), str(output))
    (output / "data.yaml").write_text(data, encoding="utf-8")
    (output / "frames.json").write_text(json.dumps(kept, indent=1), encoding="utf-8")
    return len(kept)


def thin_train(source: Path, output: Path, *, every: int, step: int) -> tuple[int, int]:
    """Copy a dataset keeping every `every`-th training frame (by frame number / `step`).

    Validation frames are copied unchanged so scores stay comparable with the full training set.
    Returns (training frames kept, validation frames copied).
    """
    if every < 2 or step < 1:
        raise ReplayError("Invalid thinning request")
    record = json.loads((source / "frames.json").read_text(encoding="utf-8"))
    kept = [
        item for item in record if item["split"] == "val" or (item["frame"] // step) % every == 0
    ]
    if not any(item["split"] == "train" for item in kept):
        raise ReplayError("Thinning removes every training frame")
    for part in ("train", "val"):
        (output / "images" / part).mkdir(parents=True)
        (output / "labels" / part).mkdir(parents=True)
    for item in kept:
        name, part = f"frame-{item['frame']:06d}", item["split"]
        shutil.copy2(
            source / "images" / part / f"{name}.jpg", output / "images" / part / f"{name}.jpg"
        )
        shutil.copy2(
            source / "labels" / part / f"{name}.txt", output / "labels" / part / f"{name}.txt"
        )
    data = (source / "data.yaml").read_text(encoding="utf-8").replace(str(source), str(output))
    (output / "data.yaml").write_text(data, encoding="utf-8")
    (output / "frames.json").write_text(json.dumps(kept, indent=1), encoding="utf-8")
    train = sum(1 for item in kept if item["split"] == "train")
    return train, len(kept) - train


def build_face_only(
    source: Path,
    output: Path,
    *,
    faces: dict[int, PixelBox],
    step: int,
    block: int,
    validation_every: int,
    gap: int,
    all_validation: bool,
    size: int,
) -> tuple[int, int]:
    """A one-class (avatar_face) dataset from Vision face boxes alone, with no caption labels.

    Candidate frames are the multiples of `step` that carry a trusted face box. With
    `all_validation` every frame is validation (a recording held out of training); otherwise the
    usual guarded time-block split applies. Returns (train frames, validation frames).
    """
    if step < 1:
        raise ReplayError("Invalid step")
    candidates = [number for number in sorted(faces) if number % step == 0]
    if not candidates:
        raise ReplayError("No frame with a face box matches the step")
    if all_validation:
        train: tuple[int, ...] = ()
        validation = tuple(range(len(candidates)))
    else:
        train, validation = split_by_time_blocks(
            len(candidates), block=block, validation_every=validation_every, gap=gap
        )
    output.mkdir(mode=0o700, parents=True)
    record: list[dict[str, object]] = []
    for part, indices in (("train", train), ("val", validation)):
        (output / "images" / part).mkdir(parents=True)
        (output / "labels" / part).mkdir(parents=True)
        numbers = [candidates[i] for i in indices]
        extract(source, numbers, output / "images" / part, size=size, rate=RATE)
        for number in numbers:
            (output / "labels" / part / f"frame-{number:06d}.txt").write_text(
                yolo_row(FACE, faces[number], canvas=CANVAS) + "\n", encoding="utf-8"
            )
            record.append({"frame": number, "split": part, "captions": 0, "face": True})
    (output / "data.yaml").write_text(
        f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n  0: {CLASS_NAMES[FACE]}\n",
        encoding="utf-8",
    )
    (output / "frames.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return len(train), len(validation)


def relabel(
    source: Path,
    output: Path,
    *,
    captions: dict[int, list[PixelBox]],
    faces: dict[int, PixelBox],
) -> tuple[int, int]:
    """Rebuild a dataset's labels from newer face observations, reusing its extracted images.

    Images are hard-linked (no copy, no new decoding) and the train/validation split is kept, so
    scores stay comparable. Returns (frames, frames with a face label). A frame whose caption
    observation is gone is refused rather than silently left unlabelled.
    """
    record = json.loads((source / "frames.json").read_text(encoding="utf-8"))
    for part in ("train", "val"):
        (output / "images" / part).mkdir(parents=True)
        (output / "labels" / part).mkdir(parents=True)
    result: list[dict[str, object]] = []
    for item in record:
        number, part = int(item["frame"]), str(item["split"])
        if number not in captions:
            raise ReplayError("A frame of the dataset has no caption observation")
        name = f"frame-{number:06d}"
        os.link(source / "images" / part / f"{name}.jpg", output / "images" / part / f"{name}.jpg")
        rows = [yolo_row(CAPTION, box, canvas=CANVAS) for box in captions[number]]
        face = faces.get(number)
        if face is not None:
            rows.append(yolo_row(FACE, face, canvas=CANVAS))
        (output / "labels" / part / f"{name}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        result.append(
            {
                "frame": number,
                "split": part,
                "captions": len(captions[number]),
                "face": face is not None,
            }
        )
    data = (source / "data.yaml").read_text(encoding="utf-8").replace(str(source), str(output))
    (output / "data.yaml").write_text(data, encoding="utf-8")
    (output / "frames.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return len(result), sum(1 for row in result if row["face"])


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
    parser.add_argument("--admission", type=Path, action="append", default=[])
    parser.add_argument("--face-response", type=Path, action="append", default=[])
    parser.add_argument("--admission-list", type=Path, help="file with one admission path per line")
    parser.add_argument(
        "--face-response-list", type=Path, help="file with one response path per line"
    )
    parser.add_argument("--step", type=int, required=True, help="keep one frame per STEP frames")
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--validation-every", type=int, required=True)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit (diagnostics only)")
    parser.add_argument(
        "--face-only",
        action="store_true",
        help="one-class face dataset from the face responses alone (no admissions needed)",
    )
    parser.add_argument(
        "--all-validation", action="store_true", help="with --face-only: hold the recording out"
    )
    parser.add_argument(
        "--relabel-from",
        type=Path,
        help="reuse the images and split of this dataset; only the labels are rebuilt",
    )
    args = parser.parse_args(argv)
    admissions = args.admission + read_paths(args.admission_list)
    responses = args.face_response + read_paths(args.face_response_list)
    if args.face_only:
        if not responses:
            raise ReplayError("Face-response paths are required")
        counts = build_face_only(
            args.source, args.output, faces=load_faces(responses), step=args.step,
            block=args.block, validation_every=args.validation_every, gap=args.gap,
            all_validation=args.all_validation, size=args.size,
        )  # fmt: skip
        print(json.dumps({"train": counts[0], "val": counts[1]}))
        return 0
    if not admissions or not responses:
        raise ReplayError("Admission and face-response paths are required")
    captions, faces = load_captions(admissions), load_faces(responses)
    if args.relabel_from is not None:
        frames, faced = relabel(args.relabel_from, args.output, captions=captions, faces=faces)
        print(json.dumps({"frames": frames, "framesWithFaceLabel": faced}))
        return 0
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
