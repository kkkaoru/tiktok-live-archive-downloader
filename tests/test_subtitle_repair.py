"""Bounded retry orchestration using synthetic evidence providers."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from replay.subtitle_repair import repair_with_retries
from replay.subtitle_retry import RepairEvidence, RepairMethod


def test_passing_frame_is_not_retried() -> None:
    measure = Mock(return_value=RepairEvidence(True, 0, 0, False))
    result = repair_with_retries(
        np.full((32, 32, 3), 100, dtype=np.uint8), measure=measure, softness=0.5
    )
    assert measure.call_count == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].method is RepairMethod.GLYPH


def test_unknown_quality_still_outputs_same_family() -> None:
    measure = Mock(return_value=RepairEvidence(True))
    result = repair_with_retries(
        np.full((32, 32, 3), 100, dtype=np.uint8), measure=measure, softness=0.5
    )
    assert measure.call_count == 2
    assert len(result.attempts) == 2
    assert result.attempts[0].method is RepairMethod.GLYPH
    assert result.attempts[1].method is RepairMethod.SOFTENED
    assert result.attempts[1].decision.cosmetic_fallback
    np.testing.assert_array_equal(result.image, np.full((32, 32, 3), 100, dtype=np.uint8))


def test_expanded_success_stops_before_softening() -> None:
    measure = Mock(
        side_effect=[RepairEvidence(True, 0.2, 0, True), RepairEvidence(True, 0, 0, False)]
    )
    result = repair_with_retries(
        np.full((32, 32, 3), 100, dtype=np.uint8), measure=measure, softness=0.5
    )
    assert measure.call_count == 2
    assert result.attempts[-1].method is RepairMethod.EXPANDED


def test_invalid_final_variant_does_not_claim_output() -> None:
    measure = Mock(return_value=RepairEvidence(False))
    with pytest.raises(ValueError, match="technical validation"):
        repair_with_retries(
            np.full((32, 32, 3), 100, dtype=np.uint8), measure=measure, softness=0.5
        )
    assert measure.call_count == 3


def test_rejects_outside_support_modification() -> None:
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    changed = np.full((32, 32, 3), 90, dtype=np.uint8)
    empty = np.zeros((32, 32), dtype=np.uint8)
    with (
        patch("replay.subtitle_repair.repair_glyphs", return_value=(changed, empty)),
        pytest.raises(ValueError, match="outside allowed support"),
    ):
        repair_with_retries(image, measure=Mock(), softness=0.5)


def test_unavailable_quality_skips_work_without_changing_pixels() -> None:
    image = np.full((48, 48, 3), 100, dtype=np.uint8)
    image[20:25, 20:25] = (180, 50, 220)
    baseline = repair_with_retries(
        image, measure=Mock(return_value=RepairEvidence(True)), softness=0.5
    )
    measure = Mock(return_value=RepairEvidence(True))
    optimized = repair_with_retries(image, measure=measure, softness=0.5, quality_available=False)
    np.testing.assert_array_equal(optimized.image, baseline.image)
    np.testing.assert_array_equal(optimized.support, baseline.support)
    assert measure.call_count == 1
    assert len(optimized.attempts) == 1
    assert optimized.attempts[0].method is RepairMethod.SOFTENED
