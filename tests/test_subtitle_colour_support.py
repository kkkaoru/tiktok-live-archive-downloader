"""Colour keying adds caption-coloured pixels inside detected line boxes and band-clipped lines."""

from collections.abc import Sequence

import numpy as np
import pytest

from replay import subtitle_colour_support as support

PINK = (180, 105, 255)  # BGR of a caption-fill pink


def _roi() -> np.ndarray:
    return np.full((40, 80, 3), 200, dtype=np.uint8)


def test_pink_inside_a_box_is_keyed_with_its_outline() -> None:
    roi = _roi()
    roi[20, 40] = PINK
    mask = support.key_mask(roi, support.FrameBoxes((100, 50), ((130, 65, 150, 75),)))
    assert (int(mask[20, 40]), int(mask[21, 41]), int(mask[20, 43])) == (255, 255, 0)


def test_pink_outside_every_box_is_not_keyed() -> None:
    roi = _roi()
    roi[5, 5] = PINK
    mask = support.key_mask(roi, support.FrameBoxes((100, 50), ((130, 65, 150, 75),)))
    assert int(mask.sum()) == 0


def test_non_pink_inside_a_box_is_not_keyed() -> None:
    mask = support.key_mask(_roi(), support.FrameBoxes((0, 0), ((0, 0, 80, 40),)))
    assert int(mask.sum()) == 0


def test_box_origin_shifts_boxes_into_roi_coordinates() -> None:
    mask = support.box_mask((40, 80), (100, 50), ((120, 60, 130, 70),))
    assert (int(mask[10, 20]), int(mask[0, 0])) == (1, 0)


def test_detector_merges_inner_masks_by_maximum_in_frame_order() -> None:
    first, second = _roi(), _roi()
    second[20, 40] = PINK

    def inner(rois: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
        assert len(rois) == 2
        return tuple(np.full((40, 80), 255 if index == 0 else 0, np.uint8) for index in range(2))

    frames = [support.FrameBoxes((0, 0), ()), support.FrameBoxes((0, 0), ((30, 10, 50, 30),))]
    detector = support.ColourKeyDetector(inner, frames)
    masks = detector.detect([first, second])
    assert (int(masks[0].min()), int(masks[1][20, 40]), detector.consumed) == (255, 255, 2)


def test_detector_rejects_more_rois_than_planned_frames() -> None:
    detector = support.ColourKeyDetector(
        lambda rois: tuple(np.zeros((40, 80), np.uint8) for _ in rois),
        [support.FrameBoxes((0, 0), ())],
    )
    with pytest.raises(ValueError, match="More ROIs"):
        detector.detect([_roi(), _roi()])


def test_a_line_clipped_by_the_band_top_beside_a_recognised_box_is_keyed() -> None:
    roi = _roi()
    roi[0:4, 40:60] = PINK
    frame = support.FrameBoxes((100, 50), ((130, 75, 160, 89),))
    mask = support.key_mask(roi, frame)
    assert (int(mask[1, 50]), int(mask[0, 40]), int(mask[3, 59])) == (255, 255, 255)


def test_a_line_clipped_by_the_band_bottom_is_keyed_too() -> None:
    roi = _roi()
    roi[36:40, 40:60] = PINK
    mask = support.key_mask(roi, support.FrameBoxes((100, 50), ((130, 65, 160, 79),)))
    assert int(mask[38, 50]) == 255


def test_pink_that_does_not_touch_a_band_edge_stays_unkeyed_outside_the_boxes() -> None:
    roi = _roi()
    roi[10:14, 40:60] = PINK
    mask = support.key_mask(roi, support.FrameBoxes((100, 50), ((130, 75, 160, 89),)))
    assert int(mask.sum()) == 0


def test_a_clipped_component_taller_than_a_line_is_scene_not_caption() -> None:
    roi = _roi()
    roi[0:30, 40:60] = PINK
    mask = support.key_mask(roi, support.FrameBoxes((100, 50), ((130, 75, 160, 89),)))
    assert (int(mask[0:12].sum()), int(mask[25, 50])) == (0, 255)


def test_a_clipped_component_away_from_every_recognised_box_is_not_keyed() -> None:
    roi = _roi()
    roi[0:4, 0:10] = PINK
    mask = support.key_mask(roi, support.FrameBoxes((100, 50), ((130, 75, 160, 89),)))
    assert int(mask.sum()) == 0


def test_a_frame_without_recognised_boxes_keys_nothing_even_at_the_edge() -> None:
    roi = _roi()
    roi[0:4, 40:60] = PINK
    assert int(support.key_mask(roi, support.FrameBoxes((0, 0), ())).sum()) == 0


def _tall_roi() -> np.ndarray:
    return np.full((80, 80, 3), 200, dtype=np.uint8)


def test_a_line_crossing_out_of_the_box_within_one_line_height_is_keyed_in_full() -> None:
    roi = _tall_roi()
    roi[12:23, 40] = PINK
    mask = support.key_mask(roi, support.FrameBoxes((0, 0), ((30, 30, 50, 50),)))
    assert (int(mask[12, 40]), int(mask[22, 40])) == (255, 255)


def test_a_component_reaching_past_one_line_height_is_scene_not_caption() -> None:
    roi = _tall_roi()
    roi[2:23, 40] = PINK
    mask = support.key_mask(roi, support.FrameBoxes((0, 0), ((30, 30, 50, 50),)))
    assert int(mask[12, 40]) == 0


def test_a_component_not_touching_any_box_is_not_keyed_by_crossing() -> None:
    roi = _tall_roi()
    roi[12:19, 70] = PINK
    mask = support.crossing_lines(roi, support.FrameBoxes((0, 0), ((30, 30, 50, 50),)))
    assert int(mask.sum()) == 0


def test_crossing_keys_nothing_without_recognised_boxes() -> None:
    roi = _tall_roi()
    roi[12:23, 40] = PINK
    assert int(support.crossing_lines(roi, support.FrameBoxes((0, 0), ())).sum()) == 0


def _scattered(seed: int) -> np.ndarray:
    """A 60x200 ROI with many caption-pink blobs, some cut by the top and bottom rows."""
    rng = np.random.default_rng(seed)
    roi = np.full((60, 200, 3), 90, dtype=np.uint8)
    for _ in range(40):
        y, x = int(rng.integers(-3, 58)), int(rng.integers(0, 195))
        roi[max(0, y) : y + int(rng.integers(2, 9)), x : x + int(rng.integers(2, 12))] = PINK
    return roi


def test_line_selection_equals_a_per_component_scan() -> None:
    import cv2

    frame = support.FrameBoxes((0, 0), ((40, 20, 120, 34), (60, 36, 150, 48)))
    roi = _scattered(3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support.pink(roi), connectivity=8)
    crossing = np.zeros(labels.shape, np.uint8)
    inside = support.box_mask(roi.shape[:2], frame.origin, frame.boxes)
    for label in range(1, count):
        member = labels == label
        rows = np.flatnonzero(member.any(axis=1))
        if inside[member].any() and rows.min() >= 6 and rows.max() < 62:
            crossing[member] = 1
    assert np.array_equal(support.crossing_lines(roi, frame), crossing)
    assert int(support.clipped_lines(roi, frame).sum()) == int(
        sum(
            stats[label, 4]
            for label in range(1, count)
            if (stats[label, 1] == 0 or stats[label, 1] + stats[label, 3] == 60)
            and stats[label, 3] <= 14
            and max(
                max(0, min(stats[label, 0] + stats[label, 2], end) - max(stats[label, 0], start))
                for start, end in ((30, 130), (50, 160))
            )
            >= 0.5 * stats[label, 2]
        )
    )


def test_shared_components_give_the_same_lines() -> None:
    frame = support.FrameBoxes((0, 0), ((40, 20, 120, 34),))
    roi = _scattered(5)
    found = support.components(support.pink(roi))
    assert np.array_equal(
        support.clipped_lines(roi, frame, found), support.clipped_lines(roi, frame)
    )
    assert np.array_equal(
        support.crossing_lines(roi, frame, found), support.crossing_lines(roi, frame)
    )
