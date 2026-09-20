"""Exact native membership and external mask validation."""

import pytest

from replay.subtitle_plan import number, repair_spans
from replay.subtitle_stream import SubtitleSpan


def test_coalesces_half_open_masks() -> None:
    assert repair_spans(
        [
            {
                "startSeconds": 0,
                "endSeconds": 2 / 30,
                "region": {"x": 8, "y": 8, "width": 16, "height": 16},
            },
            {
                "startSeconds": 2 / 30,
                "endSeconds": 4 / 30,
                "region": {"x": 8, "y": 8, "width": 16, "height": 16},
            },
        ],
        frames=5,
        width=64,
        height=64,
    ) == (SubtitleSpan(0, 4, 8, 8, 16, 16),)


def test_overlaps_union_and_gap_preserved() -> None:
    assert repair_spans(
        [
            {
                "startSeconds": 0,
                "endSeconds": 1 / 30,
                "region": {"x": 0, "y": 0, "width": 16, "height": 16},
            },
            {
                "startSeconds": 0,
                "endSeconds": 1 / 30,
                "region": {"x": 8, "y": 8, "width": 16, "height": 16},
            },
            {
                "startSeconds": 2 / 30,
                "endSeconds": 3 / 30,
                "region": {"x": 0, "y": 0, "width": 16, "height": 16},
            },
        ],
        frames=4,
        width=64,
        height=64,
    ) == (SubtitleSpan(0, 1, 0, 0, 24, 24), SubtitleSpan(2, 3, 0, 0, 16, 16))


@pytest.mark.parametrize("value", [None, True, "1", float("nan")])
def test_invalid_number(value: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        number(value)


def test_invalid_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        repair_spans([], frames=0, width=64, height=64)


def test_invalid_list() -> None:
    with pytest.raises(ValueError, match="list"):
        repair_spans(None, frames=4, width=64, height=64)


def test_invalid_object() -> None:
    with pytest.raises(ValueError, match="object"):
        repair_spans([{}], frames=4, width=64, height=64)


def test_nonintegral_pixels_refused() -> None:
    with pytest.raises(ValueError, match="Nonintegral"):
        repair_spans(
            [
                {
                    "startSeconds": 0,
                    "endSeconds": 1,
                    "region": {"x": 0.5, "y": 0, "width": 16, "height": 16},
                }
            ],
            frames=4,
            width=64,
            height=64,
        )


def test_out_of_canvas_refused() -> None:
    with pytest.raises(ValueError, match="bounds"):
        repair_spans(
            [
                {
                    "startSeconds": 0,
                    "endSeconds": 1,
                    "region": {"x": 60, "y": 0, "width": 16, "height": 16},
                }
            ],
            frames=4,
            width=64,
            height=64,
        )
