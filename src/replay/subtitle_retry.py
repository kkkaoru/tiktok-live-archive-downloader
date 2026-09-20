"""Bounded local repair decisions; cosmetic heuristics are not perceptual proof."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class RepairMethod(StrEnum):
    GLYPH = "glyph"
    EXPANDED = "expanded-glyph"
    SOFTENED = "glyph-local-softening"


@dataclass(frozen=True, slots=True)
class RepairEvidence:
    """Fractions are measured on lossless ROIs, before delivery encoding.

    Residual fraction uses the original detected glyph area as denominator;
    instability uses repaired area. None means not measured, never zero.
    Technical validity includes decode, expected clocks and permitted ROI.
    """

    technically_valid: bool
    residual_fraction: float | None = None
    unstable_fraction: float | None = None
    readable_text: bool | None = None

    def __post_init__(self) -> None:
        for value in (self.residual_fraction, self.unstable_fraction):
            if value is not None and (not isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Repair fractions must be finite and normalized")


@dataclass(frozen=True, slots=True)
class RepairDecision:
    accepted: bool
    next_method: RepairMethod | None
    reason: str
    cosmetic_fallback: bool = False


def decide_repair(method: RepairMethod, evidence: RepairEvidence) -> RepairDecision:
    """Advance twice within the same glyph-repair family, never to a blur band.

    The final variant softens the repair boundary and its immediate vicinity;
    its strength must be temporally ramped by the renderer, not abruptly switched.
    Callers must not catch fallback errors as success or restart from glyph.
    Unavailable enhancement metrics conservatively advance, not block export.
    This function neither renders nor verifies evidence supplied by callers.
    """
    if method is RepairMethod.SOFTENED:
        if not evidence.technically_valid:
            raise ValueError("Fallback failed technical validation; preserve prior delivery")
        return RepairDecision(True, None, "Validated glyph repair with local softening", True)
    next_method = {
        RepairMethod.GLYPH: RepairMethod.EXPANDED,
        RepairMethod.EXPANDED: RepairMethod.SOFTENED,
    }[method]
    if not evidence.technically_valid:
        return RepairDecision(False, next_method, "Enhancement failed technical validation")
    if (
        evidence.residual_fraction is None
        or evidence.unstable_fraction is None
        or evidence.readable_text is None
    ):
        return RepairDecision(
            False, RepairMethod.SOFTENED, "Enhancement quality evidence unavailable"
        )
    if evidence.readable_text or evidence.residual_fraction > 0.005:
        return RepairDecision(False, next_method, "Residual subtitle detected")
    if evidence.unstable_fraction > 0.02:
        return RepairDecision(False, next_method, "Repair instability exceeds tolerance")
    return RepairDecision(True, None, "Enhancement meets measured heuristic thresholds")
