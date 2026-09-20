"""Synthetic retry policy tests, not real subtitle quality verification."""

import pytest

from replay.subtitle_retry import (
    RepairDecision,
    RepairEvidence,
    RepairMethod,
    decide_repair,
)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_fraction(value: float) -> None:
    with pytest.raises(ValueError, match="finite and normalized"):
        RepairEvidence(True, residual_fraction=value)


def test_accepts_thresholds_without_retry() -> None:
    assert decide_repair(
        RepairMethod.GLYPH, RepairEvidence(True, 0.005, 0.02, False)
    ) == RepairDecision(True, None, "Enhancement meets measured heuristic thresholds")


@pytest.mark.parametrize("method", [RepairMethod.GLYPH, RepairMethod.EXPANDED])
def test_accepts_good_enhancement(method: RepairMethod) -> None:
    assert decide_repair(method, RepairEvidence(True, 0, 0, False)).accepted


def test_residual_retries_only_expanded() -> None:
    assert decide_repair(
        RepairMethod.GLYPH, RepairEvidence(True, 0.006, 0, False)
    ) == RepairDecision(False, RepairMethod.EXPANDED, "Residual subtitle detected")


def test_readable_text_disqualifies_zero_color_residual() -> None:
    assert decide_repair(RepairMethod.EXPANDED, RepairEvidence(True, 0, 0, True)) == RepairDecision(
        False, RepairMethod.SOFTENED, "Residual subtitle detected"
    )


def test_instability_advances_to_local_softening() -> None:
    assert decide_repair(
        RepairMethod.EXPANDED, RepairEvidence(True, 0, 0.021, False)
    ) == RepairDecision(False, RepairMethod.SOFTENED, "Repair instability exceeds tolerance")


@pytest.mark.parametrize(
    "evidence",
    [
        RepairEvidence(True),
        RepairEvidence(True, 0, None, False),
        RepairEvidence(True, 0, 0, None),
    ],
)
def test_missing_evidence_falls_back(evidence: RepairEvidence) -> None:
    assert decide_repair(RepairMethod.EXPANDED, evidence) == RepairDecision(
        False, RepairMethod.SOFTENED, "Enhancement quality evidence unavailable"
    )


def test_failed_enhancement_can_fall_back() -> None:
    assert decide_repair(RepairMethod.EXPANDED, RepairEvidence(False)) == RepairDecision(
        False, RepairMethod.SOFTENED, "Enhancement failed technical validation"
    )


def test_softened_repair_cosmetics_do_not_block_export() -> None:
    assert decide_repair(RepairMethod.SOFTENED, RepairEvidence(True, 1, 1, True)) == RepairDecision(
        True, None, "Validated glyph repair with local softening", True
    )


def test_softened_repair_technical_failure_is_not_a_delivery() -> None:
    with pytest.raises(ValueError, match="Fallback failed technical validation"):
        decide_repair(RepairMethod.SOFTENED, RepairEvidence(False))
