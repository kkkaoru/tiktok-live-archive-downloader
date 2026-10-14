"""Actual context tracking is connected to the repair producer, not approval."""

from fractions import Fraction
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image
from replay.subtitle_quality_observe import OriginalContext
from replay.subtitle_repair import repair_registered_context
from replay.subtitle_retry import RepairEvidence
from replay.subtitle_stream import SubtitleSpan, repair_stream


def test_registered_translation_and_occluded_donors() -> None:
    clean = np.random.default_rng(32).integers(20, 220, (256, 256, 3), dtype=np.uint8)
    clean = np.asarray(cv2.GaussianBlur(clean, (5, 5), 0), np.uint8)
    mask = np.zeros((256, 256), np.bool_)
    mask[120:124, 120:124] = True
    dirty = clean.copy()
    dirty[mask] = 255
    target = OriginalContext(dirty, mask, Fraction(1))
    clear = np.zeros_like(mask)
    before = OriginalContext(
        np.asarray(
            cv2.warpAffine(clean, np.array([[1, 0, 2], [0, 1, 0]], np.float32), (256, 256)),
            np.uint8,
        ),
        clear,
        Fraction(0),
    )
    after = OriginalContext(
        np.asarray(
            cv2.warpAffine(clean, np.array([[1, 0, -2], [0, 1, 0]], np.float32), (256, 256)),
            np.uint8,
        ),
        clear,
        Fraction(2),
    )
    result = repair_registered_context(target, donors=(before, after))
    assert np.count_nonzero(result.support) > 0
    assert result.attempts == ()
    np.testing.assert_allclose(
        result.image[result.support != 0], clean[result.support != 0], atol=2
    )
    np.testing.assert_array_equal(result.image[~mask], dirty[~mask])
    blocked = repair_registered_context(
        target, donors=(before, OriginalContext(after.image, np.ones_like(mask), Fraction(2)))
    )
    assert np.count_nonzero(blocked.support) == 0
    np.testing.assert_array_equal(blocked.unsupported, mask.astype(np.uint8) * 255)
    np.testing.assert_array_equal(blocked.image, dirty)


def test_deformation_or_hidden_occlusion_is_not_assumed_restored() -> None:
    clean = np.random.default_rng(12).integers(20, 220, (256, 256, 3), dtype=np.uint8)
    clean = np.asarray(cv2.GaussianBlur(clean, (5, 5), 0), np.uint8)
    mask = np.zeros((256, 256), np.bool_)
    mask[120:124, 120:124] = True
    dirty = clean.copy()
    dirty[mask] = 255
    rotation = cv2.getRotationMatrix2D((128, 128), 1, 1)
    moved = np.asarray(cv2.warpAffine(clean, rotation, (256, 256)), np.uint8)
    moved[116:128, 116:128] = 0
    result = repair_registered_context(
        OriginalContext(dirty, mask, Fraction(1)),
        donors=(
            OriginalContext(clean, np.zeros_like(mask), Fraction(0)),
            OriginalContext(moved, np.zeros_like(mask), Fraction(2)),
        ),
    )
    assert np.count_nonzero(result.support) == 0
    np.testing.assert_array_equal(result.image, dirty)


@pytest.mark.parametrize("batch", [True, False])
def test_real_stream_preserves_pts_unknown_and_writes_evidence(batch: bool) -> None:
    frame = np.full((64, 64, 3), 50, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[30:34, 30:34] = 255
    frames: list[Image] = []
    evidence, receipt = Mock(), Mock()
    count = repair_stream(
        (frame,) * 5,
        spans=(SubtitleSpan(0, 5, 0, 0, 64, 64),),
        expected_frames=5,
        source_clock=VideoFrameClock((120, 124, 127, 130, 134), Fraction(1, 10), 3),
        context_frames=1,
        measure=Mock(return_value=RepairEvidence(True)),
        write=frames.append,
        receipt=receipt,
        evidence=evidence,
        detect=None if batch else Mock(return_value=mask),
        detect_batch=Mock(return_value=(mask,)) if batch else None,
    )
    assert count == 5
    assert receipt.call_count == 5
    assert evidence.call_count == 5
    np.testing.assert_array_equal(frames, np.stack((frame,) * 5))
    assert np.count_nonzero(receipt.call_args.args[2].unsupported) == 64


def test_sparse_review_retains_unreviewed_frames_and_unknown_donors() -> None:
    frame = np.full((64, 64, 3), 50, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[30:34, 30:34] = 255
    write, receipt, detector = Mock(), Mock(), Mock(return_value=mask)
    assert (
        repair_stream(
            (frame,) * 5,
            spans=(SubtitleSpan(2, 3, 0, 0, 64, 64),),
            expected_frames=5,
            source_clock=VideoFrameClock((120, 124, 127, 130, 134), Fraction(1, 10), 3),
            context_frames=1,
            context_donors_verified=False,
            measure=Mock(),
            write=write,
            receipt=receipt,
            detect=detector,
        )
        == 5
    )
    assert write.call_count == 5
    assert detector.call_count == 1
    assert receipt.call_count == 1
    assert receipt.call_args.args[0] == 2
    assert np.count_nonzero(receipt.call_args.args[2].unsupported) == 64
    np.testing.assert_array_equal(write.call_args.args[0], frame)


@pytest.mark.parametrize("fault", ["short", "excess", "mask", "count", "clock", "radius"])
def test_temporal_stream_rejects_invalid_inputs(fault: str) -> None:
    frame = np.zeros((64, 64, 3), np.uint8)
    detector = Mock(return_value=(np.zeros((64, 64), np.uint8),))
    if fault == "mask":
        detector.return_value = (np.ones((64, 64), np.uint8),)
    if fault == "count":
        detector.return_value = ()
    with pytest.raises(ValueError):
        repair_stream(
            (frame,) * (1 if fault == "short" else 4 if fault == "excess" else 3),
            spans=(SubtitleSpan(0, 3, 0, 0, 64, 64),),
            expected_frames=3,
            source_clock=None if fault == "clock" else VideoFrameClock((0, 1, 2), Fraction(1), 1),
            context_frames=9 if fault == "radius" else 1,
            measure=Mock(),
            write=Mock(),
            receipt=Mock(),
            detect_batch=detector,
        )
