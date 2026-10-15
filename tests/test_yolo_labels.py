"""YOLO label geometry is normalized, bounded by the canvas and split by time blocks."""

import pytest

from replay import yolo_labels as labels
from replay.core import ReplayError
from replay.yolo_labels import PixelBox

CANVAS = (1080, 1920)


def test_a_face_box_becomes_a_normalized_center_row() -> None:
    row = labels.yolo_row(labels.FACE, PixelBox(540.0, 960.0, 216.0, 384.0), canvas=CANVAS)
    assert row == "0 0.600000 0.600000 0.200000 0.200000"


def test_a_caption_box_uses_its_own_class() -> None:
    row = labels.yolo_row(labels.CAPTION, PixelBox(0.0, 0.0, 1080.0, 192.0), canvas=CANVAS)
    assert row == "1 0.500000 0.050000 1.000000 0.100000"


def test_a_box_that_leaves_the_canvas_is_rejected() -> None:
    with pytest.raises(ReplayError, match="leaves the canvas"):
        labels.yolo_row(labels.FACE, PixelBox(1000.0, 100.0, 100.0, 100.0), canvas=CANVAS)
    with pytest.raises(ReplayError, match="leaves the canvas"):
        labels.yolo_row(labels.FACE, PixelBox(-1.0, 100.0, 100.0, 100.0), canvas=CANVAS)


def test_invalid_labels_canvases_and_boxes_are_rejected() -> None:
    with pytest.raises(ReplayError, match="Invalid YOLO label or canvas"):
        labels.yolo_row(2, PixelBox(0.0, 0.0, 10.0, 10.0), canvas=CANVAS)
    with pytest.raises(ReplayError, match="Invalid YOLO label or canvas"):
        labels.yolo_row(labels.FACE, PixelBox(0.0, 0.0, 10.0, 10.0), canvas=(0, 1920))
    with pytest.raises(ReplayError, match="Invalid YOLO box"):
        labels.yolo_row(labels.FACE, PixelBox(0.0, 0.0, 0.0, 10.0), canvas=CANVAS)
    with pytest.raises(ReplayError, match="Invalid YOLO box"):
        labels.yolo_row(labels.FACE, PixelBox(float("nan"), 0.0, 10.0, 10.0), canvas=CANVAS)


def test_clipping_keeps_only_the_visible_part() -> None:
    clipped = labels.clip_to_canvas(PixelBox(1000.0, -20.0, 200.0, 100.0), canvas=CANVAS)
    assert clipped == PixelBox(1000.0, 0.0, 80.0, 80.0)


def test_a_box_wholly_outside_the_canvas_has_nothing_visible() -> None:
    assert labels.clip_to_canvas(PixelBox(1100.0, 0.0, 50.0, 50.0), canvas=CANVAS) is None
    assert labels.clip_to_canvas(PixelBox(-60.0, 0.0, 50.0, 50.0), canvas=CANVAS) is None


def test_a_cleanup_line_box_becomes_a_pixel_box() -> None:
    assert labels.line_box([452, 1347, 577, 1384]) == PixelBox(452.0, 1347.0, 125.0, 37.0)
    with pytest.raises(ReplayError, match="left, top, right, bottom"):
        labels.line_box([1, 2, 3])


def test_every_nth_block_is_validation_and_the_rest_is_training() -> None:
    train, validation = labels.split_by_time_blocks(12, block=2, validation_every=3, gap=0)
    assert validation == (4, 5, 10, 11)
    assert train == (0, 1, 2, 3, 6, 7, 8, 9)


def test_the_guard_gap_drops_training_frames_next_to_validation() -> None:
    train, validation = labels.split_by_time_blocks(30, block=5, validation_every=3, gap=2)
    assert validation == (10, 11, 12, 13, 14, 25, 26, 27, 28, 29)
    assert train == (0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 22)


def test_invalid_split_requests_are_rejected() -> None:
    with pytest.raises(ReplayError, match="Invalid time-block split"):
        labels.split_by_time_blocks(0, block=2, validation_every=3, gap=0)
    with pytest.raises(ReplayError, match="Invalid time-block split"):
        labels.split_by_time_blocks(10, block=0, validation_every=3, gap=0)
    with pytest.raises(ReplayError, match="Invalid time-block split"):
        labels.split_by_time_blocks(10, block=2, validation_every=1, gap=0)
    with pytest.raises(ReplayError, match="Invalid time-block split"):
        labels.split_by_time_blocks(10, block=2, validation_every=3, gap=-1)
