"""YOLO caption boxes become per-frame search regions with the detector-region contract."""

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import subtitle_yolo_regions as regions
from replay.core import ReplayError
from replay.subtitle_clock import VideoFrameClock

ENVELOPE = (100, 1200, 880, 200)


def test_boxes_touching_the_envelope_become_one_padded_canvas_region() -> None:
    boxes = [(100.0, 720.0, 300.0, 740.0), (120.0, 745.0, 280.0, 760.0), (10.0, 10.0, 50.0, 30.0)]
    assert regions.region_of(boxes, scale=1.6875, envelope=ENVELOPE, canvas=(1080, 1920)) == (
        162,
        1209,
        513,
        1289,
    )


def test_no_box_in_the_envelope_rows_gives_no_region() -> None:
    boxes = [(10.0, 10.0, 50.0, 30.0)]
    assert regions.region_of(boxes, scale=1.6875, envelope=ENVELOPE, canvas=(1080, 1920)) is None


def test_padding_is_clipped_to_the_canvas() -> None:
    boxes = [(0.0, 700.0, 640.0, 740.0)]
    assert regions.region_of(boxes, scale=1.6875, envelope=ENVELOPE, canvas=(1080, 1920)) == (
        0,
        1175,
        1080,
        1255,
    )


def frames(count: int) -> list[NDArray[np.uint8]]:
    return [np.zeros((4, 4, 3), dtype=np.uint8)] * count


def test_boxes_are_detected_in_batches_and_scaled_to_the_canvas() -> None:
    sizes: list[int] = []

    def detector(images: list[NDArray[np.uint8]]) -> list[list[tuple[float, float, float, float]]]:
        sizes.append(len(images))
        return [[(100.0, 720.0, 300.0, 740.0)]] * len(images)

    found = regions.chunk_boxes(iter(frames(40)), detector, count=40, scale=1.6875)
    assert sizes == [32, 8]
    assert found[39] == [(168.75, 1215.0, 506.25, 1248.75)]


def test_a_short_decode_is_refused() -> None:
    with pytest.raises(ReplayError, match="every chunk frame"):
        regions.chunk_boxes(iter(frames(3)), lambda images: [[]] * len(images), count=4, scale=1.0)


def test_regions_are_derived_per_frame_from_canvas_boxes() -> None:
    boxes = [[(168.75, 1215.0, 506.25, 1248.75)], []]
    assert regions.yolo_regions(boxes, canvas=(1080, 1920), envelope=ENVELOPE) == [
        (162, 1209, 513, 1255),
        None,
    ]


def test_the_canvas_scale_maps_the_training_width_to_the_canvas() -> None:
    assert regions.canvas_scale((1080, 1920)) == 1.6875


def test_a_box_mask_fills_the_padded_boxes_inside_the_roi() -> None:
    roi = np.zeros((20, 30, 3), dtype=np.uint8)
    frame = regions.PlannedBoxes(
        (100, 200), ((110.0, 210.0, 115.0, 212.0), (500.0, 500.0, 510.0, 510.0))
    )
    mask = regions.box_mask(roi, frame)
    assert mask.dtype == np.uint8
    assert int(mask.sum() // 255) == 13 * 10
    assert mask[6, 6] == 255 and mask[0, 0] == 0 and mask[19, 29] == 0


def test_box_masks_are_consumed_in_render_order() -> None:
    roi = np.zeros((20, 30, 3), dtype=np.uint8)
    first = regions.PlannedBoxes((0, 0), ((0.0, 0.0, 2.0, 2.0),))
    second = regions.PlannedBoxes((0, 0), ())
    detector = regions.BoxMaskDetector([first, second])
    masks = detector.detect([roi])
    assert detector.consumed == 1 and int(masks[0].sum() // 255) == 36
    assert int(detector.detect([roi])[0].sum()) == 0
    with pytest.raises(ValueError, match="More ROIs than planned"):
        detector.detect([roi])


def test_only_caption_class_boxes_are_returned() -> None:
    boxes = SimpleNamespace(
        cls=SimpleNamespace(tolist=lambda: [0.0, 1.0]),
        xyxy=SimpleNamespace(tolist=lambda: [[1, 2, 3, 4], [5, 6, 7, 8]]),
    )
    assert regions.caption_boxes([SimpleNamespace(boxes=boxes)]) == [[(5.0, 6.0, 7.0, 8.0)]]


def test_chunk_frames_decode_the_chunk_at_the_training_width(tmp_path: Path) -> None:
    source = tmp_path / "s.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "color=c=red:s=320x240:r=30:d=2", "-pix_fmt", "yuv420p", str(source)],
        check=True, timeout=60,
    )  # fmt: skip
    clock = VideoFrameClock(tuple(range(15000, 15000 + 3000 * 5, 3000)), Fraction(1, 90000), 3000)
    decoded = list(regions.chunk_frames(source, clock, canvas=(320, 240), ffmpeg="ffmpeg"))
    assert len(decoded) == 5
    assert decoded[0].shape == (480, 640, 3)


def test_a_failing_decode_is_refused(tmp_path: Path) -> None:
    clock = VideoFrameClock((0,), Fraction(1, 90000), 3000)
    with pytest.raises(ReplayError, match="ffmpeg failed"):
        list(
            regions.chunk_frames(tmp_path / "absent.mp4", clock, canvas=(320, 240), ffmpeg="ffmpeg")
        )


def test_settings_are_read_from_the_chunk_configuration() -> None:
    config = {"kind": "yolo", "weights": "w.pt", "sha256": "a" * 64, "configDirectory": "c",
              "work": "w", "imgsz": 960}  # fmt: skip
    assert regions.settings_from(config) == regions.YoloSettings(
        Path("w.pt"), "a" * 64, Path("c"), Path("w"), 960
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"kind": "ppocr"}, "yolo detector description"),
        ({"kind": "yolo", "weights": "w.pt"}, "needs weights"),
        ({"kind": "yolo", "weights": "w", "sha256": "a", "configDirectory": "c", "work": "w",
          "imgsz": 8}, "imgsz or masks is invalid"),
        ({"kind": "yolo", "weights": "w", "sha256": "a", "configDirectory": "c", "work": "w",
          "masks": "yes"}, "imgsz or masks is invalid"),
    ],
)  # fmt: skip
def test_invalid_settings_are_refused(config: object, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        regions.settings_from(config)


class FakeModel:
    def __init__(self, weights: str) -> None:
        self.weights = weights

    def predict(self, images: list[NDArray[np.uint8]], **options: object) -> list[SimpleNamespace]:
        boxes = SimpleNamespace(
            cls=SimpleNamespace(tolist=lambda: [1.0]),
            xyxy=SimpleNamespace(tolist=lambda: [[options["imgsz"], 0, 1, 1]]),
        )
        return [SimpleNamespace(boxes=boxes) for _ in images]


def test_the_pinned_checkpoint_is_loaded_as_a_batch_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    config = tmp_path / "config"
    (config / "Ultralytics").mkdir(parents=True)
    settings: dict[str, object] = {name: False for name in ("sync", "clearml", "comet", "dvc",
                                   "mlflow", "raytune", "wandb", "tensorboard")}  # fmt: skip
    settings |= {"datasets_dir": str(tmp_path), "runs_dir": str(tmp_path)}
    (config / "Ultralytics" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr("ultralytics.models.yolo.model.YOLO", FakeModel)
    detect = regions.load_detector(
        regions.YoloSettings(
            weights, hashlib.sha256(b"weights").hexdigest(), config, tmp_path, 960, "cpu"
        )
    )
    assert detect(frames(2)) == [[(960.0, 0.0, 1.0, 1.0)], [(960.0, 0.0, 1.0, 1.0)]]


def test_chunk_frames_follow_the_clock_across_a_timestamp_gap(tmp_path: Path) -> None:
    """Frames 5.. are shown two seconds late; a chunk starting at frame 5's PTS starts there."""
    source = tmp_path / "gap.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=30:duration=1", "-vf",
         "setpts='if(gte(N,5),(N+60)/30/TB,N/30/TB)'", "-fps_mode", "passthrough",
         "-video_track_timescale", "90000", "-pix_fmt", "yuv420p", str(source)],
        check=True, timeout=60,
    )  # fmt: skip
    pts = [n * 3000 for n in range(5)] + [(n + 60) * 3000 for n in range(5, 30)]
    reference = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-vf",
         "select=eq(n\\,7),scale=640:480", "-fps_mode", "passthrough", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True, timeout=60,
    ).stdout  # fmt: skip
    clock = VideoFrameClock(tuple(pts[5:15]), Fraction(1, 90000), 3000)
    decoded = list(regions.chunk_frames(source, clock, canvas=(320, 240), ffmpeg="ffmpeg"))
    assert len(decoded) == 10
    expected = np.frombuffer(reference, dtype=np.uint8).reshape(480, 640, 3)
    assert float(np.abs(decoded[2].astype(int) - expected.astype(int)).mean()) < 1.0


def line_image() -> NDArray[np.uint8]:
    image = np.full((200, 640, 3), 90, dtype=np.uint8)
    for x in range(20, 620, 12):
        image[100:114, x : x + 8] = (180, 80, 255)  # pink glyphs across the frame
    image[60:74, 300:308] = (180, 80, 255)  # a glyph of another line above
    return image


def test_a_box_grows_along_its_own_lines_glyphs_only() -> None:
    assert regions.extend_to_glyphs(line_image(), (200.0, 98.0, 400.0, 116.0)) == (
        20.0,
        98.0,
        616.0,
        116.0,
    )


def test_a_box_stops_at_a_gap_wider_than_one_and_a_half_lines() -> None:
    image = line_image()
    image[100:114, 420:500] = 90
    grown = regions.extend_to_glyphs(image, (200.0, 98.0, 400.0, 116.0))
    assert grown == (20.0, 98.0, 420.0, 116.0)


def test_an_empty_box_is_kept() -> None:
    assert regions.extend_to_glyphs(line_image(), (10.0, 300.0, 20.0, 310.0)) == (
        10.0,
        300.0,
        20.0,
        310.0,
    )
