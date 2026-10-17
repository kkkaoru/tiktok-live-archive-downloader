"""The probe summarizes detections honestly and refuses unsafe paths before loading a model."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from replay import yolo_probe as probe
from replay.core import ReplayError
from replay.yolo_labels import CAPTION, FACE
from replay.yolo_probe import Detection

BOX = (1.0, 2.0, 3.0, 4.0)


class Tensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def tolist(self) -> list[object]:
        return self.values


class Boxes:
    def __init__(self, labels: list[int], confidences: list[float]) -> None:
        self._cls = Tensor(list(labels))
        self._conf = Tensor(list(confidences))
        self._xyxy = Tensor([[1.0, 2.0, 3.0, 4.0] for _ in labels])

    @property
    def cls(self) -> Tensor:
        return self._cls

    @property
    def conf(self) -> Tensor:
        return self._conf

    @property
    def xyxy(self) -> Tensor:
        return self._xyxy


def test_a_frame_without_boxes_has_no_detections() -> None:
    assert probe.detections_from(None) == []


def test_boxes_become_typed_detections() -> None:
    assert probe.detections_from(Boxes([0, 1], [0.9, 0.5])) == [
        Detection(FACE, 0.9, BOX),
        Detection(CAPTION, 0.5, BOX),
    ]


def test_the_summary_counts_frames_with_each_class_and_the_best_face_per_frame() -> None:
    frames = [
        [Detection(FACE, 0.9, BOX), Detection(FACE, 0.4, BOX), Detection(CAPTION, 0.7, BOX)],
        [Detection(CAPTION, 0.6, BOX)],
        [Detection(FACE, 0.5, BOX)],
        [],
    ]
    summary = probe.summarize(frames)
    assert summary["frames"] == 4
    assert summary["framesWithFace"] == 2 and summary["faceShare"] == 0.5
    assert summary["framesWithCaption"] == 2 and summary["captionShare"] == 0.5
    assert summary["faceBoxCount"] == 3
    assert summary["bestFaceConfidencePerFrameMean"] == pytest.approx(0.7)


def test_a_summary_without_any_face_has_no_mean_confidence() -> None:
    summary = probe.summarize([[Detection(CAPTION, 0.6, BOX)], []])
    assert summary["framesWithFace"] == 0 and summary["faceShare"] == 0
    assert summary["bestFaceConfidencePerFrameMean"] is None


def test_summarizing_no_frames_is_refused() -> None:
    with pytest.raises(ReplayError, match="No frames to summarize"):
        probe.summarize([])


def arguments(work: Path, **paths: Path) -> list[str]:
    weights = paths.get("weights", work / "w.pt")
    frames = paths.get("frames", work / "frames.json")
    images = paths.get("images", work / "images")
    output = paths.get("output", work / "out")
    return [
        str(weights), str(frames), str(images), str(output), "--work", str(work),
        "--config-directory", str(work / "c"),
    ]  # fmt: skip


def test_a_path_outside_the_work_directory_is_refused(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ReplayError, match="must stay under the work directory"):
        probe.parse_and_check(arguments(work, weights=tmp_path / "elsewhere.pt"))


def test_an_existing_output_directory_is_refused(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "out").mkdir(parents=True)
    with pytest.raises(ReplayError, match="Output directory already exists"):
        probe.parse_and_check(arguments(work))


def test_a_split_with_no_matching_frame_is_refused(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    record = [{"frame": 0, "split": "train", "captions": 1, "face": True}]
    (work / "frames.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ReplayError, match="No frame matches the requested split"):
        probe.parse_and_check(arguments(work))


class Result:
    def __init__(self, boxes: Boxes | None) -> None:
        self._boxes = boxes
        self.saved: list[str] = []

    @property
    def boxes(self) -> Boxes | None:
        return self._boxes

    def save(self, *, filename: str) -> str:
        self.saved.append(filename)
        return filename


class Model:
    def __init__(self, results: list[Result]) -> None:
        self.results = results
        self.calls: list[str] = []

    def predict(
        self, source: str, *, imgsz: int, device: str, conf: float, verbose: bool
    ) -> list[Result]:
        assert (imgsz, device, conf, verbose) == (640, "mps", probe.CONFIDENCE_FLOOR, False)
        self.calls.append(source)
        return [self.results[len(self.calls) - 1]]


def test_each_image_is_predicted_timed_and_only_the_first_few_are_drawn(tmp_path: Path) -> None:
    results = [Result(Boxes([0], [0.9])), Result(None), Result(Boxes([1], [0.6]))]
    model = Model(results)
    images = [tmp_path / f"frame-{n:06d}.jpg" for n in (0, 30, 60)]
    per_frame, seconds = probe.run_frames(
        model, images, tmp_path / "out", imgsz=640, device="mps", draw=2
    )
    assert per_frame == [[Detection(FACE, 0.9, BOX)], [], [Detection(CAPTION, 0.6, BOX)]]
    assert len(seconds) == 3 and all(s >= 0 for s in seconds)
    assert model.calls == [str(path) for path in images]
    assert results[0].saved == [str(tmp_path / "out" / "frame-000000.jpg")]
    assert results[1].saved == [str(tmp_path / "out" / "frame-000030.jpg")]
    assert results[2].saved == []


def test_a_report_writes_the_summary_with_speed_and_the_chosen_split(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    record = [
        {"frame": 0, "split": "val", "captions": 1, "face": False},
        {"frame": 30, "split": "val", "captions": 1, "face": True},
        {"frame": 60, "split": "train", "captions": 1, "face": False},
    ]
    (work / "frames.json").write_text(json.dumps(record), encoding="utf-8")
    args, chosen = probe.parse_and_check([*arguments(work), "--faceless-only", "--draw", "1"])
    assert [item["frame"] for item in chosen] == [0]
    model = Model([Result(Boxes([0], [0.8]))])
    summary = probe.report(model, args, chosen)
    assert summary["frames"] == 1 and summary["framesWithFace"] == 1
    assert summary["facelessOnly"] is True and summary["split"] == "val"
    assert summary["device"] == "mps" and summary["imgsz"] == 640
    assert float(str(summary["msPerFrameMedianWarm"])) >= 0
    written = json.loads((work / "out" / "summary.json").read_text())
    assert written["faceShare"] == 1.0


class Expected(Result):
    def __init__(self) -> None:
        super().__init__(None)


class Predictor:
    def __init__(self, found: list[object]) -> None:
        self.found = found

    def predict(
        self, source: str, *, imgsz: int, device: str, conf: float, verbose: bool
    ) -> list[object]:
        return self.found


def test_the_adapter_passes_only_results_of_the_expected_class() -> None:
    result = Expected()
    adapter = probe.Adapter(Predictor([result]), Expected)
    passed = adapter.predict("x.jpg", imgsz=640, device="mps", conf=0.25, verbose=False)
    assert len(passed) == 1 and passed[0] is result


def test_the_adapter_refuses_an_empty_or_foreign_prediction() -> None:
    for found in ([], [object()], [Expected(), object()]):
        adapter = probe.Adapter(Predictor(found), Expected)
        with pytest.raises(ReplayError, match="Unexpected prediction type"):
            adapter.predict("x.jpg", imgsz=640, device="mps", conf=0.25, verbose=False)


def test_main_runs_the_report_through_the_loaded_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    record = [{"frame": 0, "split": "val", "captions": 1, "face": True}]
    (work / "frames.json").write_text(json.dumps(record), encoding="utf-8")
    loaded: list[str] = []

    def yolo(weights: str) -> Predictor:
        loaded.append(weights)
        return Predictor([Result(Boxes([0], [0.8]))])

    monkeypatch.setitem(sys.modules, "ultralytics.engine.results", SimpleNamespace(Results=Result))
    monkeypatch.setitem(sys.modules, "ultralytics.models.yolo.model", SimpleNamespace(YOLO=yolo))
    monkeypatch.setenv("YOLO_CONFIG_DIR", "unset")
    monkeypatch.setenv("ULTRALYTICS_OFFLINE", "unset")
    assert probe.main(arguments(work)) == 0
    assert len(loaded) == 1
    assert json.loads(capsys.readouterr().out)["framesWithFace"] == 1
    assert os.environ["ULTRALYTICS_OFFLINE"] == "1"
