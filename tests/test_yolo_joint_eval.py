"""The joint evaluation scores faces and captions with one set of gates per recording."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from replay import yolo_joint_eval as evaluation
from replay.core import ReplayError

PINK = (180, 80, 255)


def frame_with_line() -> np.ndarray:
    image = np.full((1138, 640, 3), 90, dtype=np.uint8)
    cv2.rectangle(image, (200, 800), (400, 813), PINK, -1)
    return image


def test_iou_of_identical_disjoint_and_degenerate_boxes() -> None:
    assert evaluation.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert evaluation.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert evaluation.iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_labels_are_read_in_image_pixels(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.5 0.25\n1 0.25 0.75 0.5 0.1\n", encoding="utf-8")
    assert evaluation.read_labels(path, width=100, height=200) == (
        [(25.0, 75.0, 75.0, 125.0)],
        [(0.0, 140.0, 50.0, 160.0)],
    )


def test_box_mask_marks_the_covered_pixels() -> None:
    mask = evaluation.box_mask([(1.0, 1.0, 3.0, 2.0)], (4, 4))
    assert mask.tolist() == [
        [False, False, False, False],
        [False, True, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ]


def test_a_perfect_frame_passes_every_gate() -> None:
    tally = evaluation.Tally()
    errors: list[float] = []
    prediction = evaluation.Prediction(
        ((((100.0, 102.0, 300.0, 300.0), 0.9),)), ((198.0, 798.0, 403.0, 816.0),)
    )
    labels = ([(100.0, 100.0, 300.0, 300.0)], [(198.0, 798.0, 403.0, 816.0)])
    evaluation.score_frame(tally, frame_with_line(), labels, prediction, errors, scale=1.5)
    assert errors == [3.0]
    assert evaluation.report(tally, errors, 12.0) == {
        "face": {
            "frames": 1,
            "detectionRate": 1.0,
            "topErrorCanvasPx": {"p50": 3.0, "p90": 3.0},
        },
        "caption": {
            "glyphCoverage": 1.0,
            "lines": 1,
            "lineRecallIoU50": 1.0,
            "predictedBoxes": 1,
            "falseBoxShare": 0.0,
        },
        "msPerFrame": 12.0,
        "pass": {"face": True, "caption": True, "speed": True, "joint": True},
    }


def test_a_missed_face_a_false_box_and_a_slow_model_fail() -> None:
    tally = evaluation.Tally()
    errors: list[float] = []
    prediction = evaluation.Prediction((), ((10.0, 10.0, 50.0, 30.0),))
    labels = ([(100.0, 100.0, 300.0, 300.0)], [(198.0, 798.0, 403.0, 816.0)])
    evaluation.score_frame(tally, frame_with_line(), labels, prediction, errors, scale=1.5)
    assert evaluation.report(tally, errors, 90.0) == {
        "face": {
            "frames": 1,
            "detectionRate": 0.0,
            "topErrorCanvasPx": {"p50": None, "p90": None},
        },
        "caption": {
            "glyphCoverage": 0.0,
            "lines": 1,
            "lineRecallIoU50": 0.0,
            "predictedBoxes": 1,
            "falseBoxShare": 1.0,
        },
        "msPerFrame": 90.0,
        "pass": {"face": False, "caption": False, "speed": False, "joint": False},
    }


def write_frame(directory: Path, name: str) -> Path:
    (directory / "images").mkdir(parents=True, exist_ok=True)
    (directory / "labels").mkdir(parents=True, exist_ok=True)
    path = directory / "images" / f"{name}.jpg"
    cv2.imwrite(str(path), frame_with_line())
    (directory / "labels" / f"{name}.txt").write_text(
        "0 0.3125 0.17575 0.3125 0.17575\n1 0.4703125 0.7108963 0.3203125 0.0158172\n",
        encoding="utf-8",
    )
    return path


def test_evaluate_batches_the_images_and_reports_the_recording(tmp_path: Path) -> None:
    files = [write_frame(tmp_path, "a"), write_frame(tmp_path, "b"), write_frame(tmp_path, "c")]
    batches: list[int] = []

    def detector(images: list[np.ndarray]) -> list[evaluation.Prediction]:
        batches.append(len(images))
        return [
            evaluation.Prediction(
                (((100.0, 100.0, 300.0, 300.0), 0.9),), ((198.0, 798.0, 403.0, 816.0),)
            )
        ] * len(images)

    result = evaluation.evaluate(detector, files, tmp_path / "labels", canvas_width=1080, batch=2)
    assert batches == [2, 1]
    assert result["frames"] == 3
    assert result["pass"] == {"face": True, "caption": True, "speed": True, "joint": True}


def test_evaluate_refuses_no_images_a_bad_batch_and_an_unreadable_image(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="No images"):
        evaluation.evaluate(lambda images: [], [], tmp_path, canvas_width=1080, batch=2)
    with pytest.raises(ReplayError, match="invalid batch"):
        evaluation.evaluate(lambda images: [], [tmp_path], tmp_path, canvas_width=1080, batch=0)
    with pytest.raises(ReplayError, match="could not be read"):
        evaluation.read_image(tmp_path / "absent.jpg")


def test_ultralytics_results_become_boxes_per_class() -> None:
    boxes = SimpleNamespace(
        cls=SimpleNamespace(tolist=lambda: [0.0, 1.0]),
        conf=SimpleNamespace(tolist=lambda: [0.8, 0.6]),
        xyxy=SimpleNamespace(tolist=lambda: [[1, 2, 3, 4], [5, 6, 7, 8]]),
    )
    assert evaluation.predictions_from([SimpleNamespace(boxes=boxes)]) == [
        evaluation.Prediction((((1.0, 2.0, 3.0, 4.0), 0.8),), ((5.0, 6.0, 7.0, 8.0),))
    ]


def test_main_refuses_an_existing_output(tmp_path: Path) -> None:
    (tmp_path / "out.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="overwrite"):
        evaluation.main(
            ["w.pt", str(tmp_path), str(tmp_path / "out.json"), "--recording", "a",
             "--config-directory", "c", "--work", "w"]
        )  # fmt: skip


class FakeModel:
    def __init__(self, weights: str) -> None:
        self.weights = weights

    def predict(self, images: list[np.ndarray], **options: object) -> list[SimpleNamespace]:
        boxes = SimpleNamespace(
            cls=SimpleNamespace(tolist=lambda: [0.0, 1.0]),
            conf=SimpleNamespace(tolist=lambda: [0.9, 0.9]),
            xyxy=SimpleNamespace(tolist=lambda: [[100, 100, 300, 300], [198, 798, 403, 816]]),
        )
        return [SimpleNamespace(boxes=boxes) for _ in images]


def test_main_writes_the_document_per_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "work" / "dataset"
    write_frame(dataset / "x", "a")
    (dataset / "images" / "rec").mkdir(parents=True)
    (dataset / "labels" / "rec").mkdir(parents=True)
    (dataset / "x" / "images" / "a.jpg").rename(dataset / "images" / "rec" / "a.jpg")
    (dataset / "x" / "labels" / "a.txt").rename(dataset / "labels" / "rec" / "a.txt")
    config = tmp_path / "config"
    (config / "Ultralytics").mkdir(parents=True)
    services = ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "wandb", "tensorboard")
    settings: dict[str, object] = dict.fromkeys(services, False)
    settings |= {"datasets_dir": str(tmp_path / "work"), "runs_dir": str(tmp_path / "work")}
    (config / "Ultralytics" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr("ultralytics.models.yolo.model.YOLO", FakeModel)
    output = tmp_path / "work" / "eval.json"
    code = evaluation.main(
        ["w.pt", str(dataset), str(output), "--recording", "rec", "--config-directory",
         str(config), "--work", str(tmp_path / "work"), "--device", "cpu"]
    )  # fmt: skip
    assert code == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["recordings"]["rec"]["frames"] == 1
    assert document["recordings"]["rec"]["pass"]["caption"] is True
