"""Classical ROI-only glyph repair; no model, cloud, or rectangle replacement."""

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class InpaintSettings:
    expansion: int = 2
    softness: float = 0.0

    def __post_init__(self) -> None:
        if not 1 <= self.expansion <= 6:
            raise ValueError("Glyph expansion must be between 1 and 6 pixels")
        if not isfinite(self.softness) or not 0 <= self.softness <= 1:
            raise ValueError("Softness must be finite and normalized")


def validate_roi(image: Image) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 BGR image")
    if min(image.shape[:2]) < 8 or image.shape[0] * image.shape[1] > 2_000_000:
        raise ValueError("Invalid or excessive subtitle ROI")


def pink_glyph_mask(image: Image, *, expansion: int = 2) -> Image:
    """Profile-specific pink subtitle seeds, closed to include white interiors.

    Only pass an independently localized subtitle ROI. This color profile is
    not a universal subtitle detector and must not scan unrelated image areas.
    """
    validate_roi(image)
    InpaintSettings(expansion=expansion)
    colors = image.astype(np.int16)
    seeds = (colors[:, :, 2] - colors[:, :, 1] > 24) & (colors[:, :, 0] - colors[:, :, 1] > 8)
    mask = seeds.astype(np.uint8) * 255
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    return np.asarray(
        cv2.dilate(closed, np.ones((3, 3), dtype=np.uint8), iterations=expansion),
        dtype=np.uint8,
    )


def repair_glyphs(image: Image, *, settings: InpaintSettings) -> tuple[Image, Image]:
    """Use the same Telea repair for every variant; soften only local support.

    Returns the repaired BGR ROI and allowed modification support. Renderer
    callers must ramp softness in time; this spatial kernel does not establish
    temporal consistency or visual acceptance on its own.
    """
    mask = pink_glyph_mask(image, expansion=settings.expansion)
    if not np.any(mask):
        return image.copy(), mask
    repaired = np.asarray(cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA), dtype=np.uint8)
    if settings.softness == 0:
        return repaired, mask
    support = np.asarray(
        cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=3), dtype=np.uint8
    )
    feather = np.asarray(cv2.GaussianBlur(mask, (7, 7), 0), dtype=np.float32) / 255
    softened = np.asarray(cv2.GaussianBlur(repaired, (7, 7), 0), dtype=np.float32)
    alpha = (feather * settings.softness)[:, :, None]
    mixed = np.rint(repaired * (1 - alpha) + softened * alpha).clip(0, 255)
    result = mixed.astype(np.uint8)
    result[support == 0] = image[support == 0]
    return result, support


def ramp_softness(*, frame: int, frames: int, target: float, ramp_frames: int = 6) -> float:
    """Smoothstep at interval boundaries; callers provide frame-local clocks."""
    if frames < 1 or not 0 <= frame < frames or ramp_frames < 1:
        raise ValueError("Invalid softness ramp clocks")
    if not isfinite(target) or not 0 <= target <= 1:
        raise ValueError("Invalid target softness")
    distance = min(frame, frames - 1 - frame)
    fraction = min(1.0, distance / ramp_frames)
    return target * fraction * fraction * (3 - 2 * fraction)
