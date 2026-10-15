"""Dataset building keeps only genuine observations and records the frames lacking a face label."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from replay import yolo_dataset as dataset
from replay.core import ReplayError
from replay.yolo_labels import PixelBox


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_caption_boxes_are_clipped_line_boxes() -> None:
    frame: dict[str, object] = {"lineBoxes": [[100, 200, 300, 260], [1000, 1900, 1200, 2000]]}
    assert dataset.caption_boxes(frame) == [
        PixelBox(100.0, 200.0, 200.0, 60.0),
        PixelBox(1000.0, 1900.0, 80.0, 20.0),
    ]


def test_a_frame_without_line_boxes_has_no_caption_boxes() -> None:
    assert dataset.caption_boxes({}) == []
    assert dataset.caption_boxes({"lineBoxes": None}) == []


def test_a_line_box_wholly_outside_the_canvas_is_dropped() -> None:
    assert dataset.caption_boxes({"lineBoxes": [[1100, 10, 1200, 50]]}) == []


def face(method: str, confidence: float) -> dict[str, object]:
    return {
        "method": method,
        "faces": [
            {
                "region": {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0},
                "confidence": confidence,
            }
        ],
    }


def test_a_detection_is_a_face_label_whatever_its_confidence() -> None:
    assert dataset.face_box(face("detection", 0.58)) == PixelBox(10.0, 20.0, 30.0, 40.0)


def test_tracking_is_a_label_only_at_high_confidence() -> None:
    assert dataset.face_box(face("tracking", 0.9)) == PixelBox(10.0, 20.0, 30.0, 40.0)
    assert dataset.face_box(face("tracking", 0.89)) is None


def test_missing_frames_and_ambiguous_faces_have_no_label() -> None:
    assert dataset.face_box({"method": "missing", "faces": []}) is None
    one = {"region": {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}, "confidence": 0.9}
    assert dataset.face_box({"method": "detection", "faces": [one, one]}) is None


def test_captions_use_genuine_observations_only_and_number_frames_across_chunks(
    tmp_path: Path,
) -> None:
    first = write(
        tmp_path / "a.json",
        {
            "frames": [
                {
                    "sampled": True,
                    "classification": "overlay-subtitle",
                    "lineBoxes": [[1, 2, 11, 12]],
                },
                {
                    "sampled": False,
                    "classification": "overlay-subtitle",
                    "lineBoxes": [[1, 2, 11, 12]],
                },
            ]
        },
    )
    second = write(
        tmp_path / "b.json",
        {
            "frames": [
                {"sampled": True, "classification": "unknown", "lineBoxes": [[1, 2, 11, 12]]},
                {
                    "sampled": True,
                    "classification": "overlay-subtitle",
                    "lineBoxes": [[5, 6, 15, 16]],
                },
            ]
        },
    )
    assert dataset.load_captions([first, second]) == {
        0: [PixelBox(1.0, 2.0, 10.0, 10.0)],
        3: [PixelBox(5.0, 6.0, 10.0, 10.0)],
    }


def test_faces_are_numbered_across_windows(tmp_path: Path) -> None:
    first = write(tmp_path / "a.json", [face("detection", 0.6), {"method": "missing", "faces": []}])
    second = write(tmp_path / "b.json", [face("tracking", 0.95)])
    assert dataset.load_faces([first, second]) == {
        0: PixelBox(10.0, 20.0, 30.0, 40.0),
        2: PixelBox(10.0, 20.0, 30.0, 40.0),
    }


def run_cli(
    tmp_path: Path, admission: Path, response: Path, *, step: str
) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "source.mp4"
    if not source.exists():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=1080x1920:r=30:d=4",
             "-pix_fmt", "yuv420p", str(source)],
            check=True,
        )  # fmt: skip
    command = [
        sys.executable, "-m", "replay.yolo_dataset", str(source), str(tmp_path / "out"),
        "--admission", str(admission), "--face-response", str(response),
        "--step", step, "--block", "1", "--validation-every", "2", "--gap", "0", "--size", "270",
    ]  # fmt: skip
    return subprocess.run(command, capture_output=True, text=True, check=False)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_the_command_line_writes_split_images_labels_and_a_record(tmp_path: Path) -> None:
    frames = [
        {
            "sampled": True,
            "classification": "overlay-subtitle",
            "lineBoxes": [[100, 1500, 900, 1600]],
        }
    ] * 120
    admission = write(tmp_path / "admission.json", {"frames": frames})
    response = write(tmp_path / "response.json", [face("detection", 0.6)] * 120)
    result = run_cli(tmp_path, admission, response, step="30")
    assert result.returncode == 0, result.stderr
    output = tmp_path / "out"
    record = json.loads((output / "frames.json").read_text())
    assert [(r["frame"], r["split"], r["face"]) for r in record] == [
        (0, "train", True),
        (60, "train", True),
        (30, "val", True),
        (90, "val", True),
    ]
    assert (output / "images/train/frame-000000.jpg").is_file()
    assert (output / "labels/val/frame-000030.txt").read_text().splitlines() == [
        "1 0.462963 0.807292 0.740741 0.052083",
        "0 0.023148 0.020833 0.027778 0.020833",
    ]
    assert (output / "data.yaml").read_text() == (
        f"path: {output}\ntrain: images/train\nval: images/val\nnames:\n"
        "  0: avatar_face\n  1: source_caption\n"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_the_command_line_rejects_a_step_that_matches_no_observed_frame(tmp_path: Path) -> None:
    observed = {
        "sampled": True,
        "classification": "overlay-subtitle",
        "lineBoxes": [[100, 1500, 900, 1600]],
    }
    unobserved = {"sampled": False, "classification": "overlay-subtitle"}
    admission = write(tmp_path / "admission.json", {"frames": [unobserved, observed, observed]})
    response = write(tmp_path / "response.json", [face("detection", 0.6)] * 3)
    result = run_cli(tmp_path, admission, response, step="30")
    assert result.returncode != 0
    assert "No observed subtitle frame matches the step" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_main_builds_the_dataset_in_process(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=1080x1920:r=30:d=4",
         "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )  # fmt: skip
    observed = {
        "sampled": True,
        "classification": "overlay-subtitle",
        "lineBoxes": [[100, 1500, 900, 1600]],
    }
    admission = write(tmp_path / "admission.json", {"frames": [observed] * 120})
    response = write(
        tmp_path / "response.json", [face("detection", 0.6)] * 60 + [face("tracking", 0.5)] * 60
    )
    arguments = [
        str(source), str(tmp_path / "out"), "--admission", str(admission),
        "--face-response", str(response), "--step", "30", "--block", "1",
        "--validation-every", "2", "--gap", "0", "--size", "270", "--limit", "3",
    ]  # fmt: skip
    assert dataset.main(arguments) == 0
    record = json.loads((tmp_path / "out/frames.json").read_text())
    assert [(r["frame"], r["split"], r["face"]) for r in record] == [
        (0, "train", True),
        (60, "train", False),
        (30, "val", True),
    ]
    assert (tmp_path / "out/labels/train/frame-000060.txt").read_text().splitlines() == [
        "1 0.462963 0.807292 0.740741 0.052083"
    ]


def test_path_lists_keep_their_order_and_ignore_blank_lines(tmp_path: Path) -> None:
    listing = tmp_path / "paths.txt"
    listing.write_text("/a/one.json\n\n/b/two.json\n", encoding="utf-8")
    assert dataset.read_paths(listing) == [Path("/a/one.json"), Path("/b/two.json")]
    assert dataset.read_paths(None) == []


def test_main_requires_admission_and_face_paths(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="paths are required"):
        dataset.main(
            [str(tmp_path / "s.mp4"), str(tmp_path / "o"), "--step", "30", "--block", "1",
             "--validation-every", "2", "--gap", "0"]
        )  # fmt: skip


def test_a_faced_dataset_keeps_only_frames_with_a_face_label(tmp_path: Path) -> None:
    source = tmp_path / "all"
    for part in ("train", "val"):
        (source / "images" / part).mkdir(parents=True)
        (source / "labels" / part).mkdir(parents=True)
    rows = [
        {"frame": 0, "split": "train", "captions": 1, "face": True},
        {"frame": 30, "split": "train", "captions": 1, "face": False},
        {"frame": 60, "split": "val", "captions": 1, "face": True},
    ]
    for row in rows:
        name = f"frame-{row['frame']:06d}"
        (source / "images" / str(row["split"]) / f"{name}.jpg").write_bytes(b"jpeg")
        (source / "labels" / str(row["split"]) / f"{name}.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    (source / "frames.json").write_text(json.dumps(rows), encoding="utf-8")
    (source / "data.yaml").write_text(f"path: {source}\ntrain: images/train\n", encoding="utf-8")
    output = tmp_path / "faced"
    assert dataset.select_faced(source, output) == 2
    assert sorted(p.name for p in (output / "images/train").iterdir()) == ["frame-000000.jpg"]
    assert sorted(p.name for p in (output / "labels/val").iterdir()) == ["frame-000060.txt"]
    assert (output / "data.yaml").read_text() == f"path: {output}\ntrain: images/train\n"
    assert [r["frame"] for r in json.loads((output / "frames.json").read_text())] == [0, 60]


def test_a_faced_dataset_needs_at_least_one_face_label(tmp_path: Path) -> None:
    source = tmp_path / "all"
    source.mkdir()
    rows = [{"frame": 0, "split": "train", "captions": 1, "face": False}]
    (source / "frames.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ReplayError, match="No frame carries a trusted face label"):
        dataset.select_faced(source, tmp_path / "faced")


def make_dataset(root: Path, frames: list[tuple[int, str]]) -> Path:
    source = root / "all"
    for part in ("train", "val"):
        (source / "images" / part).mkdir(parents=True)
        (source / "labels" / part).mkdir(parents=True)
    rows = []
    for number, split in frames:
        name = f"frame-{number:06d}"
        (source / "images" / split / f"{name}.jpg").write_bytes(b"jpeg")
        (source / "labels" / split / f"{name}.txt").write_text("1 0.5 0.5 0.1 0.1\n")
        rows.append({"frame": number, "split": split, "captions": 1, "face": True})
    (source / "frames.json").write_text(json.dumps(rows), encoding="utf-8")
    (source / "data.yaml").write_text(f"path: {source}\ntrain: images/train\n", encoding="utf-8")
    return source


def test_thinning_keeps_every_nth_training_frame_and_all_validation_frames(tmp_path: Path) -> None:
    frames = [
        (0, "train"),
        (60, "train"),
        (120, "train"),
        (180, "train"),
        (240, "val"),
        (300, "val"),
    ]
    source = make_dataset(tmp_path, frames)
    output = tmp_path / "thin"
    assert dataset.thin_train(source, output, every=2, step=60) == (2, 2)
    assert sorted(p.name for p in (output / "images/train").iterdir()) == [
        "frame-000000.jpg",
        "frame-000120.jpg",
    ]
    assert sorted(p.name for p in (output / "labels/val").iterdir()) == [
        "frame-000240.txt",
        "frame-000300.txt",
    ]
    assert (output / "data.yaml").read_text() == f"path: {output}\ntrain: images/train\n"
    assert [r["frame"] for r in json.loads((output / "frames.json").read_text())] == [
        0,
        120,
        240,
        300,
    ]


def test_an_invalid_thinning_request_is_refused(tmp_path: Path) -> None:
    source = make_dataset(tmp_path, [(0, "train"), (60, "val")])
    with pytest.raises(ReplayError, match="Invalid thinning request"):
        dataset.thin_train(source, tmp_path / "a", every=1, step=60)
    with pytest.raises(ReplayError, match="Invalid thinning request"):
        dataset.thin_train(source, tmp_path / "b", every=2, step=0)


def test_thinning_that_removes_every_training_frame_is_refused(tmp_path: Path) -> None:
    source = make_dataset(tmp_path, [(60, "train"), (120, "val")])
    with pytest.raises(ReplayError, match="removes every training frame"):
        dataset.thin_train(source, tmp_path / "thin", every=2, step=60)
