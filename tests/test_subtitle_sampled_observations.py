"""Sampled observations fill unread frames conservatively and label themselves as filled."""

import numpy as np
import pytest

from replay import subtitle_sampled_observations as sampled
from replay.subtitle_sampled_observations import Box

LEFT = Box(0, 0, 10, 10)
RIGHT = Box(20, 0, 10, 10)


def test_sample_indices_cover_first_stride_last_and_triggers() -> None:
    assert sampled.sample_indices(10, 4, triggers=(6,)) == (0, 4, 6, 8, 9)


def test_sample_indices_reject_bad_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        sampled.sample_indices(0, 4)
    with pytest.raises(ValueError, match="outside"):
        sampled.sample_indices(5, 2, triggers=(5,))


def test_fill_uses_bracketing_union_and_keeps_samples() -> None:
    filled = sampled.fill(5, {0: (LEFT,), 4: (RIGHT,)})
    assert filled[0] == (LEFT,)
    assert filled[2] == (LEFT, RIGHT)
    assert filled[4] == (RIGHT,)


def test_fill_requires_first_and_last_samples() -> None:
    with pytest.raises(ValueError, match="first and last"):
        sampled.fill(5, {1: (LEFT,), 4: (RIGHT,)})


def test_coverage_counts_missed_line_and_extra_mask() -> None:
    truth = ((LEFT,), (LEFT, RIGHT), (LEFT,))
    predicted = ((LEFT,), (LEFT,), (LEFT, Box(50, 50, 10, 10)))
    result = sampled.coverage(truth, predicted)
    assert result == sampled.Coverage(total=4, missed=1, true_area=400.0, extra_area=100.0)


def test_coverage_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="different"):
        sampled.coverage(((LEFT,),), ())


def test_receipt_declares_filled_not_raw() -> None:
    assert sampled.receipt(5, 4, (0, 4), ()) == {
        "version": 1,
        "kind": "sampled-ocr-observation",
        "rawOcr": False,
        "interpolated": True,
        "fillRule": "bracketing-sample-union-v1",
        "frames": 5,
        "stride": 4,
        "sampledFrames": [0, 4],
        "triggerFrames": [],
        "qualityApproved": False,
    }


def test_change_triggers_bracket_an_abrupt_change() -> None:
    bands = np.zeros((8, 2, 2), dtype=np.uint8)
    bands[:, 0, 0] = [0, 1, 0, 1, 0, 1, 0, 1]
    bands[5:] += 100
    assert sampled.change_triggers(bands) == (4, 5)


def test_change_triggers_ignore_a_perfectly_static_interval() -> None:
    assert sampled.change_triggers(np.zeros((6, 2, 2), dtype=np.uint8)) == ()


def test_change_triggers_reject_too_few_frames() -> None:
    with pytest.raises(ValueError, match="at least two"):
        sampled.change_triggers(np.zeros((1, 2, 2), dtype=np.uint8))
