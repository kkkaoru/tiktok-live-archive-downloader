"""Synthetic stream clocks and spatial isolation tests."""

from unittest.mock import Mock

import numpy as np
import pytest

from replay.subtitle_retry import RepairEvidence
from replay.subtitle_stream import SubtitleSpan, repair_stream


@pytest.mark.parametrize("start,end", [(-1, 4), (4, 4)])
def test_invalid_span_clock(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="range"):
        SubtitleSpan(start, end, 0, 0, 8, 8)


def test_invalid_span_region() -> None:
    with pytest.raises(ValueError, match="region"):
        SubtitleSpan(0, 1, -1, 0, 8, 8)


def test_stream_preserves_unmasked_frames_and_source() -> None:
    source = np.full((40, 40, 3), 100, dtype=np.uint8)
    source[15:20, 15:20] = (180, 50, 220)
    original = source.copy()
    write = Mock()
    receipt = Mock()
    assert (
        repair_stream(
            (source, source, source),
            spans=(SubtitleSpan(1, 2, 8, 8, 24, 24),),
            expected_frames=3,
            measure=Mock(return_value=RepairEvidence(True)),
            write=write,
            receipt=receipt,
        )
        == 3
    )
    assert write.call_count == 3
    assert receipt.call_count == 1
    assert receipt.call_args.args[0] == 1
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(write.call_args_list[0].args[0], original)
    np.testing.assert_array_equal(write.call_args_list[2].args[0], original)
    np.testing.assert_array_equal(write.call_args_list[1].args[0][:8], original[:8])


def test_overlapping_spans_refused() -> None:
    with pytest.raises(ValueError, match="Overlapping"):
        repair_stream(
            (),
            spans=(SubtitleSpan(0, 2, 0, 0, 8, 8), SubtitleSpan(1, 3, 0, 0, 8, 8)),
            expected_frames=3,
            measure=Mock(),
            write=Mock(),
            receipt=Mock(),
        )


def test_too_few_frames_refused() -> None:
    with pytest.raises(ValueError, match="before expected"):
        repair_stream((), spans=(), expected_frames=1, measure=Mock(), write=Mock(), receipt=Mock())


def test_too_many_frames_refused() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="excess"):
        repair_stream(
            (frame, frame),
            spans=(),
            expected_frames=1,
            measure=Mock(),
            write=Mock(),
            receipt=Mock(),
        )


def test_out_of_canvas_refused() -> None:
    with pytest.raises(ValueError, match="canvas"):
        repair_stream(
            (np.zeros((16, 16, 3), dtype=np.uint8),),
            spans=(SubtitleSpan(0, 1, 12, 12, 8, 8),),
            expected_frames=1,
            measure=Mock(),
            write=Mock(),
            receipt=Mock(),
        )


def test_invalid_channels_refused() -> None:
    with pytest.raises(ValueError, match="BGR"):
        repair_stream(
            (np.zeros((16, 16), dtype=np.uint8),),
            spans=(SubtitleSpan(0, 1, 0, 0, 8, 8),),
            expected_frames=1,
            measure=Mock(),
            write=Mock(),
            receipt=Mock(),
        )
