"""The review overlay draws the detector's boxes and streams every frame to the encoder."""

import hashlib
import json
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from replay import yolo_overlay as overlay
from replay.core import ReplayError
from replay.yolo_joint_eval import Prediction


def blank() -> np.ndarray:
    return np.zeros((64, 48, 3), dtype=np.uint8)


def result(labels: list[float], scores: list[float], corners: list[list[float]]) -> SimpleNamespace:
    boxes = SimpleNamespace(
        cls=SimpleNamespace(tolist=lambda: labels),
        conf=SimpleNamespace(tolist=lambda: scores),
        xyxy=SimpleNamespace(tolist=lambda: corners),
    )
    return SimpleNamespace(boxes=boxes)


def test_drawing_marks_the_face_the_captions_and_the_header_on_a_copy() -> None:
    image = blank()
    prediction = Prediction((((10.0, 30.0, 30.0, 50.0), 0.9),), ((5.0, 55.0, 40.0, 60.0),))
    drawn = overlay.draw(image, prediction, caption_scores=[0.8], header="x")
    assert image.sum() == 0
    assert tuple(drawn[40, 10]) == overlay.FACE_COLOUR
    assert tuple(drawn[57, 5]) == overlay.CAPTION_COLOUR
    assert tuple(drawn[30, 0]) == overlay.FACE_COLOUR


def test_caption_scores_keep_only_caption_confidences() -> None:
    found = [result([0.0, 1.0, 1.0], [0.9, 0.7, 0.6], [[0, 0, 1, 1]] * 3)]
    assert overlay.caption_scores(found) == [[0.7, 0.6]]


def frames(count: int) -> Iterator[tuple[int, np.ndarray]]:
    for number in range(count):
        yield number, blank()


def detect(images: list[np.ndarray]) -> list[SimpleNamespace]:
    return [result([0.0, 1.0], [0.9, 0.7], [[10, 30, 30, 50], [5, 55, 40, 60]]) for _ in images]


def test_every_frame_is_detected_in_batches_and_encoded(tmp_path: Path) -> None:
    output = tmp_path / "o.mp4"
    sink = overlay.encoder(output, size=(48, 64), rate=Fraction(30000, 1001), ffmpeg="ffmpeg")
    timing = overlay.render(frames(5), detect, sink, batch=2, label="t")
    assert timing["frames"] == 5.0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=nb_frames,width,height", "-of",
         "json", str(output)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert json.loads(probe.stdout)["streams"][0] == {"width": 48, "height": 64, "nb_frames": "5"}


def test_an_invalid_batch_and_a_failing_encoder_are_refused(tmp_path: Path) -> None:
    sink = overlay.encoder(
        tmp_path / "a.mp4", size=(48, 64), rate=Fraction(30000, 1001), ffmpeg="ffmpeg"
    )
    with pytest.raises(ReplayError, match="Invalid batch"):
        overlay.render(frames(1), detect, sink, batch=0, label="t")
    sink.kill()
    sink.wait()
    broken = subprocess.Popen(["false"], stdin=subprocess.PIPE)
    with pytest.raises(ReplayError, match="encoder failed"):
        overlay.render(iter(()), detect, broken, batch=1, label="t")


def test_an_encoder_without_input_is_refused() -> None:
    sink = subprocess.Popen(["true"])
    with pytest.raises(ReplayError, match="no input stream"):
        overlay.render(frames(1), detect, sink, batch=1, label="t")
    sink.wait()


def test_main_refuses_an_existing_output_and_an_invalid_segment(tmp_path: Path) -> None:
    existing = tmp_path / "o.mp4"
    existing.write_bytes(b"")
    common = ["--weights", "w", "--weights-sha256", "0", "--config-directory", "c", "--work", "w"]
    with pytest.raises(ReplayError, match="overwrite"):
        overlay.main(["s.mp4", str(existing), *common, "--start-seconds", "0", "--seconds", "1"])
    with pytest.raises(ReplayError, match="Invalid segment"):
        overlay.main(
            ["s.mp4", str(tmp_path / "n.mp4"), *common, "--start-seconds", "0", "--seconds", "0"]
        )


class FakeModel:
    def __init__(self, weights: str) -> None:
        self.weights = weights

    def predict(self, images: list[np.ndarray], **options: object) -> list[SimpleNamespace]:
        return detect(images)


def test_main_renders_a_segment_with_the_pinned_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "s.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "color=c=gray:s=320x240:r=30:d=1", "-pix_fmt", "yuv420p", str(source)],
        check=True, timeout=60,
    )  # fmt: skip
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    config = tmp_path / "config"
    (config / "Ultralytics").mkdir(parents=True)
    settings: dict[str, object] = {name: False for name in ("sync", "clearml", "comet", "dvc",
                                   "mlflow", "raytune", "wandb", "tensorboard")}  # fmt: skip
    settings |= {"datasets_dir": str(tmp_path), "runs_dir": str(tmp_path)}
    (config / "Ultralytics" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr("ultralytics.models.yolo.model.YOLO", FakeModel)
    digest = hashlib.sha256(b"w").hexdigest()
    output = tmp_path / "o.mp4"
    code = overlay.main(
        [str(source), str(output), "--weights", str(weights), "--weights-sha256", digest,
         "--config-directory", str(config), "--work", str(tmp_path), "--start-seconds", "0.2",
         "--seconds", "0.5", "--device", "cpu"]
    )  # fmt: skip
    assert code == 0
    assert output.stat().st_size > 0
