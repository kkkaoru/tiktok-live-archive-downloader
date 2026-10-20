"""OCR-free residual check: unchanged caption-colour pixels inside detected line boxes.

A repaired frame still carries caption text where caption-coloured pixels sit inside the
original's recognised line boxes and are (nearly) identical to the original. Scene pixels of
the same hue are excluded because they lie outside the boxes or were changed by the repair.
The count is evidence for escalation and review, never proof of absence or approval.
"""

from collections.abc import Sequence

import cv2
import numpy as np

from replay.subtitle_colour_support import pink
from replay.subtitle_inpaint import Image

GATE = 40  # unchanged caption-colour pixels per audited frame that trigger re-repair
SAME = 16  # maximum per-channel difference that still counts as unchanged
SCENE_HUES = (140, 10)  # loose pink/salmon OpenCV hue range (wraps through red)
SCENE_SATURATION, SCENE_VALUE = 40, 120
SCENE_KERNEL = 21  # an opening this wide removes every glyph stroke but keeps solid objects


def scene_pink(original: Image) -> Image:
    """0/1 mask of solid pinkish scene objects: loose pink survives a glyph-wide opening."""
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hues = (hue >= SCENE_HUES[0]) | (hue <= SCENE_HUES[1])
    loose = (hues & (saturation >= SCENE_SATURATION) & (value >= SCENE_VALUE)).astype(np.uint8)
    solid = cv2.morphologyEx(loose, cv2.MORPH_OPEN, np.ones((SCENE_KERNEL,) * 2, np.uint8))
    return np.asarray(cv2.dilate(solid, np.ones((3, 3), np.uint8)), dtype=np.uint8)


def unchanged_pink(cleaned: Image, original: Image, allowed: Image) -> int:
    """Count caption-colour pixels that are unchanged by the repair inside ``allowed`` (0/1)."""
    if cleaned.shape != original.shape or allowed.shape != cleaned.shape[:2]:
        raise ValueError("Cleaned frame, original frame and box mask must share one geometry")
    difference = np.abs(cleaned.astype(np.int16) - original.astype(np.int16)).max(axis=2)
    unchanged = np.minimum(np.minimum(pink(cleaned), pink(original)), (difference <= SAME))
    return int((np.minimum(unchanged, allowed) & (1 - scene_pink(original))).sum())


def flagged(counts: Sequence[int], *, gate: int = GATE) -> tuple[int, ...]:
    """Positions whose residual count reaches the gate."""
    if gate < 1:
        raise ValueError("The residual gate must be positive")
    return tuple(index for index, count in enumerate(counts) if count >= gate)
