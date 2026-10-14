"""Synthetic stream clocks and spatial isolation tests."""

from unittest.mock import Mock

import numpy as np
import pytest

from replay.subtitle_inpaint import Image
from replay.subtitle_retry import RepairEvidence
from replay.subtitle_stream import SubtitleSpan, repair_stream


@pytest.fixture(autouse=True)
def synthetic_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    def detect(image: Image) -> Image:
        return np.asarray(image[:, :, 2] != image[:, :, 1], dtype=np.uint8) * 255

    monkeypatch.setattr(
        "replay.subtitle_stream.vision_text_masks", lambda images: tuple(map(detect, images))
    )


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


def test_failed_writer_never_emits_lossless_evidence() -> None:
    evidence = Mock()
    with pytest.raises(OSError, match="encoder failed"):
        repair_stream(
            (np.zeros((16, 16, 3), dtype=np.uint8),),
            spans=(SubtitleSpan(0, 1, 0, 0, 16, 16),),
            expected_frames=1,
            measure=Mock(return_value=RepairEvidence(True)),
            quality_available=False,
            write=Mock(side_effect=OSError("encoder failed")),
            receipt=Mock(),
            evidence=evidence,
        )
    evidence.assert_not_called()


def test_chunked_repair_is_pixel_identical_to_whole_sequence() -> None:
    gray = ((np.indices((32, 32)).sum(axis=0) % 2) * 180 + 30).astype(np.uint8)
    source = np.repeat(gray[:, :, None], 3, axis=2)
    source[14:18, 14:18] = (180, 50, 220)
    whole: list[Image] = []
    split: list[Image] = []
    repair_stream(
        (source,) * 18,
        spans=(SubtitleSpan(0, 18, 0, 0, 32, 32),),
        expected_frames=18,
        measure=Mock(return_value=RepairEvidence(True)),
        quality_available=False,
        write=whole.append,
        receipt=Mock(),
    )
    repair_stream(
        (source,) * 9,
        spans=(SubtitleSpan(0, 9, 0, 0, 32, 32),),
        expected_frames=9,
        measure=Mock(return_value=RepairEvidence(True)),
        quality_available=False,
        write=split.append,
        receipt=Mock(),
        ramp_total_frames=18,
    )
    repair_stream(
        (source,) * 9,
        spans=(SubtitleSpan(0, 9, 0, 0, 32, 32),),
        expected_frames=9,
        measure=Mock(return_value=RepairEvidence(True)),
        quality_available=False,
        write=split.append,
        receipt=Mock(),
        ramp_start_frame=9,
        ramp_total_frames=18,
    )
    np.testing.assert_array_equal(np.stack(split), np.stack(whole))
    assert not np.array_equal(whole[8], whole[0])


def test_batch_detection_preserves_frames_and_reuses_masks_across_retries() -> None:
    frame = np.full((16, 16, 3), 100, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    detector = Mock(side_effect=[(mask,) * 7, (mask,)])
    write, receipt = Mock(), Mock()
    assert (
        repair_stream(
            (frame,) * 9,
            spans=(SubtitleSpan(1, 9, 0, 0, 16, 16),),
            expected_frames=9,
            measure=Mock(return_value=RepairEvidence(True)),
            write=write,
            receipt=receipt,
            detect_batch=detector,
        )
        == 9
    )
    assert detector.call_count == 2
    assert len(detector.call_args_list[0].args[0]) == 7
    assert len(detector.call_args_list[1].args[0]) == 1
    assert not detector.call_args_list[0].args[0][0].flags.writeable
    assert write.call_count == 9
    assert receipt.call_count == 8
    assert receipt.call_args_list[0].args[0] == 1
    assert receipt.call_args_list[-1].args[0] == 8
    assert len(receipt.call_args.args[2].attempts) == 2


def test_batch_memory_budget_and_single_detector_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("replay.subtitle_stream.MAX_BATCH_FRAME_BYTES", 768)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    detector = Mock(return_value=np.zeros((16, 16), dtype=np.uint8))
    assert (
        repair_stream(
            (frame,) * 3,
            spans=(SubtitleSpan(0, 3, 0, 0, 16, 16),),
            expected_frames=3,
            measure=Mock(return_value=RepairEvidence(True, 0, 0, False)),
            write=Mock(),
            receipt=Mock(),
            detect=detector,
        )
        == 3
    )
    assert detector.call_count == 3


def test_mismatched_batch_mask_count_fails_before_writing() -> None:
    write = Mock()
    with pytest.raises(ValueError, match="mask count"):
        repair_stream(
            (np.zeros((16, 16, 3), dtype=np.uint8),),
            spans=(SubtitleSpan(0, 1, 0, 0, 16, 16),),
            expected_frames=1,
            measure=Mock(),
            write=write,
            receipt=Mock(),
            detect_batch=Mock(return_value=()),
        )
    write.assert_not_called()


def test_conflicting_detectors_refused() -> None:
    with pytest.raises(ValueError, match="either"):
        repair_stream(
            (),
            spans=(),
            expected_frames=1,
            measure=Mock(),
            write=Mock(),
            receipt=Mock(),
            detect=Mock(),
            detect_batch=Mock(),
        )


def test_changing_batch_geometry_refused() -> None:
    with pytest.raises(ValueError, match="geometry changed"):
        repair_stream(
            (np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.uint8)),
            spans=(),
            expected_frames=2,
            measure=Mock(),
            write=Mock(),
            receipt=Mock(),
        )


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
