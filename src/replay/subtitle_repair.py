"""Bounded same-family frame repair with explicit evidence and decision receipts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from replay.subtitle_inpaint import (
    Image,
    InpaintSettings,
    registered_glyph_samples,
    repair_glyphs,
    unchanged_outside_support,
)
from replay.subtitle_retry import RepairDecision, RepairEvidence, RepairMethod, decide_repair
from replay.subtitle_vision import vision_text_mask

if TYPE_CHECKING:
    from replay.subtitle_quality_observe import OriginalContext


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
    unsupported: Image | None = None


QualityMeasure = Callable[[Image, Image, Image], RepairEvidence]
TextDetector = Callable[[Image], Image]


def repair_registered_context(
    source: OriginalContext, *, donors: tuple[OriginalContext, ...]
) -> RepairedFrame:
    """Propose only bidirectional, bracketing, agreeing original donor pixels.

    Context geometry cannot observe deformation hidden behind glyphs. Support is
    conditional sampling evidence, never independent restoration approval. Unknown
    pixels retain original bytes; there is no spatial-inpainting fallback.
    """
    # The observation layer imports stream plans; defer this shared kernel dependency.
    from replay.subtitle_quality_observe import PHOTO_CODES, OriginalContext, register

    target = source.excluded.astype(np.uint8) * 255
    before, after = source.image.copy(), source.image.copy()
    left, right = np.zeros_like(target), np.zeros_like(target)
    yy, xx = np.indices(target.shape, dtype=np.float32)
    grid = np.stack((xx, yy), axis=-1)
    for donor in donors:
        # Exclude both glyph footprints when choosing either direction's track population.
        excluded = source.excluded | donor.excluded
        fit_source = OriginalContext(source.image, excluded, source.pts)
        fit_donor = OriginalContext(donor.image, excluded, donor.pts)
        forward = register(fit_source, fit_donor)
        backward = register(fit_donor, fit_source)
        if forward.displacement is None or backward.displacement is None:
            # A rejected rigid hypothesis needs held-out local support, not weaker inliers.
            forward = register(fit_source, fit_donor, model="local-affine")
            backward = register(fit_donor, fit_source, model="local-affine")
        reverse_support = (
            cv2.remap(
                backward.context_geometry.astype(np.float32),
                forward.donor_xy[:, :, 0],
                forward.donor_xy[:, :, 1],
                cv2.INTER_LINEAR,
            )
            == 1
        )
        eligible = target.copy()
        eligible[~(forward.context_geometry & reverse_support)] = 0
        sampled, support = registered_glyph_samples(
            source.image,
            donor.image,
            source_mask=eligible,
            donor_mask=donor.excluded.astype(np.uint8) * 255,
            forward=np.asarray(forward.donor_xy - grid, dtype=np.float32),
            backward=np.asarray(backward.donor_xy - grid, dtype=np.float32),
        )
        image, covered = (before, left) if donor.pts < source.pts else (after, right)
        fresh = (support != 0) & (covered == 0)
        image[fresh] = sampled[fresh]
        covered[fresh] = 255
    agree = np.max(np.abs(before.astype(np.int16) - after.astype(np.int16)), axis=2) <= PHOTO_CODES
    support = np.asarray((left != 0) & (right != 0) & agree, dtype=np.uint8) * 255
    result = source.image.copy()
    cv2.copyTo(before, support, result)
    unsupported = np.asarray((target != 0) & (support == 0), dtype=np.uint8) * 255
    return RepairedFrame(result, support, (), unsupported)


def repair_with_retries(
    image: Image,
    *,
    measure: QualityMeasure,
    softness: float,
    quality_available: bool = True,
    detect: TextDetector | None = None,
) -> RepairedFrame:
    """Try at most three variants; unknown quality still produces local repair.

    Technical validity here means spatial integrity, not a video decode/clock
    receipt. The movie writer must additionally validate its encoded output.
    ``softness`` must be ramped by the sequence caller to avoid abrupt changes.
    The callback must not mutate its read-only source/result/support views.
    """
    source_view = image.view()
    source_view.flags.writeable = False
    text_mask = (vision_text_mask if detect is None else detect)(source_view)
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
        result, support = repair_glyphs(image, settings=config, text_mask=text_mask)
        if not unchanged_outside_support(image, result, support):
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
