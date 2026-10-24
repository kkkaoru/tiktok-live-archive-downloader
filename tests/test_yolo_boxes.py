"""Streamed caption boxes: what yolo-track writes is exactly what a cleanup chunk would compute."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import subtitle_yolo_regions as regions
from replay import yolo_boxes, yolo_track
from replay.core import ReplayError

TIME_BASE = Fraction(1, 90000)


class Tensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def tolist(self) -> list[object]:
        return self.values


class Boxes:
    def __init__(self, items: list[tuple[int, list[float]]]) -> None:
        self.cls = Tensor([item[0] for item in items])
        self.conf = Tensor([0.9 for _ in items])
        self.xyxy = Tensor([item[1] for item in items])


class Result:
    def __init__(self, items: list[tuple[int, list[float]]]) -> None:
        self.boxes = Boxes(items)


def blank() -> NDArray[np.uint8]:
    return np.zeros((40, 640, 3), dtype=np.uint8)


def clock_file(tmp_path: Path, count: int) -> Path:
    path = tmp_path / "video-clock.json"
    frames = [{"pts": 3000 * index, "duration": 3000} for index in range(count)]
    path.write_text(json.dumps({"frames": frames, "streams": [{"time_base": "1/90000"}]}))
    return path


@pytest.fixture
def writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> yolo_boxes.Writer:
    monkeypatch.setattr(yolo_track, "probe_video", lambda _s, _f: (1280, 720, 30))
    monkeypatch.setattr(yolo_track, "sha256", lambda _p: "s" * 64)
    monkeypatch.setattr(yolo_boxes, "RANGE_FRAMES", 2)
    (tmp_path / "boxes").mkdir()
    (tmp_path / "boxes" / "complete.json").write_text("{}")  # an earlier pass's marker
    return yolo_boxes.Writer(
        tmp_path / "boxes", clock_path=clock_file(tmp_path, 5), source=tmp_path / "s.mp4",
        weights_sha256="w" * 64, imgsz=640, ffprobe="ffprobe",
    )  # fmt: skip


def test_the_writer_streams_scaled_grown_caption_boxes_by_presentation_time(
    writer: yolo_boxes.Writer, tmp_path: Path
) -> None:
    directory = tmp_path / "boxes"
    assert not yolo_boxes.complete(directory)
    caption = [10.0, 5.0, 20.0, 15.0]
    writer(
        [(0, blank()), (1, blank()), (2, blank())],
        [Result([(1, caption), (0, [1.0, 1.0, 2.0, 2.0])]), Result([]), Result([(1, caption)])],
    )
    assert sorted(path.name for path in directory.iterdir()) == [
        "range-000000000000-000000003000.json"
    ]
    writer([(3, blank()), (4, blank())], [Result([]), Result([])])
    writer.close()
    assert yolo_boxes.complete(directory)
    assert json.loads((directory / "complete.json").read_text())["frames"] == 5
    expected = regions.extend_to_glyphs(blank(), (10.0, 5.0, 20.0, 15.0))
    scaled = tuple(value * 2.0 for value in expected)
    loaded = yolo_boxes.load(
        directory, [3000, 6000, 9000], time_base=TIME_BASE, source_sha256="s" * 64,
        weights_sha256="w" * 64, imgsz=640,
    )  # fmt: skip
    assert loaded == [[], [scaled], []]


def test_frames_out_of_clock_order_are_refused(writer: yolo_boxes.Writer) -> None:
    with pytest.raises(ReplayError, match="do not follow the decoded clock"):
        writer([(0, blank()), (2, blank())], [Result([]), Result([])])


def test_load_refuses_missing_frames_and_other_detections(
    writer: yolo_boxes.Writer, tmp_path: Path
) -> None:
    directory = tmp_path / "boxes"
    writer([(0, blank()), (1, blank())], [Result([]), Result([])])

    def ask(where: Path, pts: list[int], weights: str = "w" * 64) -> object:
        return yolo_boxes.load(
            where, pts, time_base=TIME_BASE, source_sha256="s" * 64, weights_sha256=weights,
            imgsz=640,
        )  # fmt: skip

    assert ask(directory, [0, 3000]) == [[], []]
    assert ask(directory, [0, 3000, 6000]) is None
    assert ask(directory, [0, 1500]) is None
    assert ask(directory, [0], "x" * 64) is None
    assert ask(directory, []) is None
    assert ask(tmp_path / "none", [0]) is None


def test_a_malformed_range_file_is_an_error(writer: yolo_boxes.Writer, tmp_path: Path) -> None:
    directory = tmp_path / "boxes"
    writer([(0, blank()), (1, blank())], [Result([]), Result([])])
    path = directory / "range-000000000000-000000003000.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "boxes": [[]]}))
    with pytest.raises(ReplayError, match="malformed"):
        yolo_boxes.load(
            directory, [0], time_base=TIME_BASE, source_sha256="s" * 64,
            weights_sha256="w" * 64, imgsz=640,
        )  # fmt: skip


def test_coverage_needs_ranges_reaching_both_ends(tmp_path: Path) -> None:
    for name in ("range-000000000000-000000000900.json", "range-000000001000-000000001900.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "other.json").write_text("{}")
    assert yolo_boxes.covered(tmp_path, 0, 1900) is True
    assert yolo_boxes.covered(tmp_path, 1000, 2000) is False
    assert yolo_boxes.covered(tmp_path, 2000, 2100) is False


def test_a_frame_clock_needs_frames_and_a_time_base(tmp_path: Path) -> None:
    assert yolo_boxes.frame_clock(clock_file(tmp_path, 2)) == ((0, 3000), TIME_BASE)
    path = tmp_path / "bad.json"
    path.write_text('{"frames": []}')
    with pytest.raises(ReplayError, match="not a decoded video clock"):
        yolo_boxes.frame_clock(path)


def test_an_oversized_range_file_is_an_error(
    writer: yolo_boxes.Writer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer([(0, blank()), (1, blank())], [Result([]), Result([])])
    monkeypatch.setattr(yolo_boxes, "MAX_RANGE_BYTES", 10)
    with pytest.raises(ReplayError, match="exceeds its byte budget"):
        yolo_boxes.load(
            tmp_path / "boxes", [0], time_base=TIME_BASE, source_sha256="s" * 64,
            weights_sha256="w" * 64, imgsz=640,
        )  # fmt: skip
