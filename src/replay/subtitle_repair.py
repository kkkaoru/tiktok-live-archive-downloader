"""Bounded same-family frame repair with explicit evidence and decision receipts."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from replay.subtitle_inpaint import Image, InpaintSettings, repair_glyphs
from replay.subtitle_retry import RepairDecision, RepairEvidence, RepairMethod, decide_repair


@dataclass(frozen=True, slots=True)
class RepairAttempt:
    method: RepairMethod
    evidence: RepairEvidence
    decision: RepairDecision


@dataclass(frozen=True, slots=True)
class RepairedFrame:
    image: Image
    support: Image
    attempts: tuple[RepairAttempt, ...]


QualityMeasure = Callable[[Image, Image, Image], RepairEvidence]


def repair_with_retries(
    image: Image,
    *,
    measure: QualityMeasure,
    softness: float,
    quality_available: bool = True,
) -> RepairedFrame:
    """Try at most three variants; unknown quality still produces local repair.

    Technical validity here means spatial integrity, not a video decode/clock
    receipt. The movie writer must additionally validate its encoded output.
    ``softness`` must be ramped by the sequence caller to avoid abrupt changes.
    The callback must not mutate its read-only source/result/support views.
    """
    settings = {
        RepairMethod.GLYPH: InpaintSettings(expansion=2),
        RepairMethod.EXPANDED: InpaintSettings(expansion=4),
        RepairMethod.SOFTENED: InpaintSettings(expansion=4, softness=softness),
    }
    attempts: list[RepairAttempt] = []
    next_method: RepairMethod | None = (
        RepairMethod.GLYPH if quality_available else RepairMethod.SOFTENED
    )
    for method, config in settings.items():
        if method is not next_method:
            continue
        result, support = repair_glyphs(image, settings=config)
        if not np.array_equal(result[support == 0], image[support == 0]):
            raise ValueError("Repair changed pixels outside allowed support")
        source_view, result_view, support_view = image.view(), result.view(), support.view()
        source_view.flags.writeable = False
        result_view.flags.writeable = False
        support_view.flags.writeable = False
        evidence = measure(source_view, result_view, support_view)
        decision = decide_repair(method, evidence)
        attempts.append(RepairAttempt(method, evidence, decision))
        if decision.accepted:
            return RepairedFrame(result, support, tuple(attempts))
        next_method = decision.next_method
    raise ValueError("Repair exhausted without a valid same-family result")
