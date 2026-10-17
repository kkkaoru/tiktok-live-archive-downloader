"""Unchanged caption colour inside detected boxes is residual; changed or outside is not."""

import numpy as np
import pytest

from replay import subtitle_colour_residual as residual

PINK = (180, 105, 255)


def _frame() -> np.ndarray:
    return np.full((20, 30, 3), 200, dtype=np.uint8)


def _inside() -> np.ndarray:
    allowed = np.zeros((20, 30), dtype=np.uint8)
    allowed[5:15, 5:25] = 1
    return allowed


def test_unchanged_pink_inside_the_boxes_is_counted() -> None:
    original = _frame()
    original[10, 10:13] = PINK
    assert residual.unchanged_pink(original.copy(), original, _inside()) == 3


def test_repaired_pink_is_not_counted() -> None:
    original = _frame()
    original[10, 10:13] = PINK
    cleaned = _frame()
    assert residual.unchanged_pink(cleaned, original, _inside()) == 0


def test_scene_pink_outside_the_boxes_is_not_counted() -> None:
    original = _frame()
    original[2, 2:6] = PINK
    assert residual.unchanged_pink(original.copy(), original, _inside()) == 0


def test_geometry_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="one geometry"):
        residual.unchanged_pink(_frame(), np.zeros((10, 10, 3), np.uint8), _inside())


def test_flagged_reports_positions_at_or_above_the_gate() -> None:
    assert residual.flagged([0, 39, 40, 500]) == (2, 3)


def test_gate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        residual.flagged([1], gate=0)
