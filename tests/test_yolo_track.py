"""Face/caption tracking: the face rule, batching, scaling, the decoded frame stream and the CLI."""

import argparse
import json
import shutil
import subprocess
import sys
import types
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import yolo_track as tracking
from replay.core import ReplayError
from replay.yolo_track import Row
from replay.yolo_train import REQUIRED_OFF


class Tensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def tolist(self) -> list[object]:
        return self.values


class Boxes:
    def __init__(self, items: list[tuple[int, float, list[float]]]) -> None:
        self.cls = Tensor([item[0] for item in items])
        self.conf = Tensor([item[1] for item in items])
        self.xyxy = Tensor([item[2] for item in items])


class Result:
    def __init__(self, boxes: Boxes | None) -> None:
        self.boxes = boxes


class FakeModel:
    """Answers each image by the value of its first pixel: 0 = nothing, 1 = one face."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def predict(
        self,
        source: list[NDArray[np.uint8]],
        *,
        imgsz: int,
        device: str,
        conf: float,
        verbose: bool,
    ) -> Sequence[Result]:
        self.batches.append(len(source))
        return [
            Result(Boxes([(0, 0.9, [10.0, 20.0, 30.0, 40.0])]) if int(image[0, 0, 0]) else None)
            for image in source
        ]


def image(value: int) -> NDArray[np.uint8]:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def frames(values: list[int]) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    return iter([(index * 30, image(value)) for index, value in enumerate(values)])


def test_no_boxes_gives_an_empty_row() -> None:
    assert tracking.row_from(7, None, scale=1.0) == Row(7, None, 0, 0)


def test_only_captions_gives_a_row_without_a_face() -> None:
    boxes = Boxes([(1, 0.8, [0.0, 0.0, 1.0, 1.0]), (1, 0.6, [0.0, 0.0, 1.0, 1.0])])
    assert tracking.row_from(3, boxes, scale=1.0) == Row(3, None, 0, 2)


def test_the_highest_confidence_face_is_chosen_and_scaled_to_source_pixels() -> None:
    boxes = Boxes(
        [
            (0, 0.4, [1.0, 2.0, 3.0, 4.0]),
            (0, 0.9, [10.0, 20.0, 30.0, 40.0]),
            (1, 0.7, [0.0, 0.0, 5.0, 5.0]),
        ]
    )
    assert tracking.row_from(9, boxes, scale=1.6875) == Row(
        9, (16.875, 33.75, 50.625, 67.5, 0.9), 2, 1
    )


def test_batches_are_filled_in_order_and_the_remainder_is_flushed() -> None:
    model = FakeModel()
    rows, seconds = tracking.track(
        frames([1, 0, 1, 1, 0]), model, scale=2.0, batch=2, device="cpu", imgsz=640
    )
    assert model.batches == [2, 2, 1]
    assert [row.frame for row in rows] == [0, 30, 60, 90, 120]
    assert [row.face is not None for row in rows] == [True, False, True, True, False]
    assert rows[0].face == (20.0, 40.0, 60.0, 80.0, 0.9)
    assert seconds >= 0.0


def test_no_frames_gives_no_rows_and_no_prediction() -> None:
    model = FakeModel()
    assert tracking.track(iter([]), model, scale=1.0, batch=4, device="cpu", imgsz=640)[0] == []
    assert model.batches == []


def test_an_invalid_batch_is_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid batch size"):
        tracking.track(iter([]), FakeModel(), scale=1.0, batch=0, device="cpu", imgsz=640)
    with pytest.raises(ReplayError, match="Invalid batch size"):
        tracking.track(iter([]), FakeModel(), scale=1.0, batch=65, device="cpu", imgsz=640)


def test_a_detector_that_drops_a_result_is_refused() -> None:
    class Short(FakeModel):
        def predict(
            self,
            source: list[NDArray[np.uint8]],
            *,
            imgsz: int,
            device: str,
            conf: float,
            verbose: bool,
        ) -> Sequence[Result]:
            return [Result(None)]

    with pytest.raises(ReplayError, match="different number of results"):
        tracking.track(frames([1, 1]), Short(), scale=1.0, batch=2, device="cpu", imgsz=640)


def test_the_adapter_returns_results_of_the_expected_class_only() -> None:
    class Predictor:
        def __init__(self, items: list[object]) -> None:
            self.items = items

        def predict(self, source: object, **options: object) -> list[object]:
            return self.items

    good = tracking.Adapter(Predictor([Result(None)]), Result)
    assert len(good.predict([image(0)], imgsz=640, device="cpu", conf=0.25, verbose=False)) == 1
    with pytest.raises(ReplayError, match="Unexpected prediction type"):
        tracking.Adapter(Predictor(["x"]), Result).predict(
            [image(0)], imgsz=640, device="cpu", conf=0.25, verbose=False
        )
    with pytest.raises(ReplayError, match="Unexpected prediction type"):
        tracking.Adapter(Predictor([]), Result).predict(
            [image(0)], imgsz=640, device="cpu", conf=0.25, verbose=False
        )


def test_the_scaled_height_matches_the_training_images() -> None:
    assert tracking.scaled_height(1080, 1920) == 1138
    assert tracking.scaled_height(640, 480) == 480
    with pytest.raises(ReplayError, match="Invalid video size"):
        tracking.scaled_height(0, 10)


def test_the_document_names_weights_video_rule_and_timing() -> None:
    rows = [Row(0, (1.0, 2.0, 3.0, 4.0, 0.5), 1, 2), Row(30, None, 0, 0)]
    result = tracking.document(
        rows,
        weights=(Path("w.pt"), "aa"),
        source=(Path("v.mp4"), "bb"),
        size=(1080, 1920),
        rate=30,
        step=30,
        timing={"wallSeconds": 2.0},
    )
    assert result["protocol"] == "yolo-face-caption-track-v1"
    assert result["provenance"] == "yolo"
    assert result["weights"] == {"path": "w.pt", "sha256": "aa"}
    assert result["source"] == {"path": "v.mp4", "sha256": "bb"}
    assert result["frames"] == [
        {"frame": 0, "face": [1.0, 2.0, 3.0, 4.0, 0.5], "faceCount": 1, "captions": 2},
        {"frame": 30, "face": None, "faceCount": 0, "captions": 0},
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for the stream test")
def test_the_decoded_stream_samples_the_requested_frames_at_the_training_width(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x480:rate=10:duration=3",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    got = list(
        tracking.video_frames(
            video, width=320, height=480, start=4, end=25, step=5, rate=10, ffmpeg="ffmpeg"
        )
    )
    assert [number for number, _ in got] == [4, 9, 14, 19, 24]
    assert got[0][1].shape == (960, 640, 3)
    assert tracking.probe_video(video, "ffprobe") == (320, 480, 10)


def test_an_invalid_range_or_step_is_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid frame range or step"):
        next(
            tracking.video_frames(
                Path("x.mp4"), width=10, height=10, start=5, end=5, step=1, rate=10, ffmpeg="ffmpeg"
            )
        )
    with pytest.raises(ReplayError, match="Invalid frame range or step"):
        next(
            tracking.video_frames(
                Path("x.mp4"), width=10, height=10, start=0, end=5, step=0, rate=10, ffmpeg="ffmpeg"
            )
        )


def test_a_failing_decoder_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="ffmpeg failed"):
        list(
            tracking.video_frames(
                tmp_path / "missing.mp4",
                width=10,
                height=10,
                start=0,
                end=5,
                step=1,
                rate=10,
                ffmpeg="ffmpeg",
            )
        )


def test_a_non_integer_frame_rate_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Done:
        stdout = json.dumps({"streams": [{"width": 4, "height": 4, "r_frame_rate": "30000/1001"}]})

    monkeypatch.setattr(subprocess, "run", lambda *args, **options: Done())
    with pytest.raises(ReplayError, match="integer frame rates"):
        tracking.probe_video(tmp_path / "x.mp4", "ffprobe")


def test_the_sha256_of_a_file_is_its_content_digest(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    assert tracking.sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for the run test")
def test_run_and_main_write_evidence_for_a_synthetic_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x480:rate=10:duration=2",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    settings = tmp_path / "config" / "Ultralytics"
    settings.mkdir(parents=True)
    work = tmp_path.resolve()
    (settings / "settings.json").write_text(
        json.dumps(
            {name: False for name in REQUIRED_OFF}
            | {"datasets_dir": str(work / "d"), "runs_dir": str(work / "r")}
        ),
        encoding="utf-8",
    )

    class Predictor:
        def __init__(self, path: str) -> None:
            self.path = path

        def predict(self, source: list[NDArray[np.uint8]], **options: object) -> list[Result]:
            return [Result(Boxes([(0, 0.8, [1.0, 2.0, 3.0, 4.0])])) for _ in source]

    results_module = types.ModuleType("ultralytics.engine.results")
    results_module.__dict__["Results"] = Result
    model_module = types.ModuleType("ultralytics.models.yolo.model")
    model_module.__dict__["YOLO"] = Predictor
    for name, module in (
        ("ultralytics", types.ModuleType("ultralytics")),
        ("ultralytics.engine", types.ModuleType("ultralytics.engine")),
        ("ultralytics.engine.results", results_module),
        ("ultralytics.models", types.ModuleType("ultralytics.models")),
        ("ultralytics.models.yolo", types.ModuleType("ultralytics.models.yolo")),
        ("ultralytics.models.yolo.model", model_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    output = tmp_path / "out.json"
    code = tracking.main(
        [str(video), str(output), "--weights", str(weights), "--weights-sha256",
         tracking.sha256(weights), "--config-directory", str(tmp_path / "config"),
         "--work", str(work), "--end", "20", "--step", "10", "--batch", "2", "--device", "cpu"]
    )  # fmt: skip
    written = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert [item["frame"] for item in written["frames"]] == [0, 10]
    assert written["frames"][0]["face"] == [0.5, 1.0, 1.5, 2.0, 0.8]
    assert written["rate"] == 10
    assert written["step"] == 10
    assert set(written["timing"]) == {
        "wallSeconds", "inferenceSeconds", "decodeAndOtherSeconds", "framesPerSecond",
    }  # fmt: skip


def test_parse_defaults(tmp_path: Path) -> None:
    args = tracking.parse(
        [str(tmp_path / "v.mp4"), str(tmp_path / "o.json"), "--weights", "w",
         "--weights-sha256", "x", "--config-directory", "c", "--work", "w", "--end", "5",
         "--step", "1"]
    )  # fmt: skip
    assert isinstance(args, argparse.Namespace)
    assert (args.start, args.batch, args.device, args.imgsz) == (0, 16, "mps", 640)


def test_main_refuses_to_overwrite_before_loading_anything(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        tracking.main(
            [str(tmp_path / "v.mp4"), str(output), "--weights", "w", "--weights-sha256", "x",
             "--config-directory", str(tmp_path), "--work", str(tmp_path), "--end", "10",
             "--step", "5"]
        )  # fmt: skip
