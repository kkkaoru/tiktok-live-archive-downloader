"""Classical ROI-only glyph repair; no model, cloud, or rectangle replacement."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
Flow = NDArray[np.float32]
MAX_ROUND_TRIP_ERROR = 0.5


@dataclass(frozen=True, slots=True)
class InpaintSettings:
    expansion: int = 2
    softness: float = 0.0
    fill_enclosed_white: bool = False
    algorithm: Literal["telea", "navier-stokes"] = "telea"

    def __post_init__(self) -> None:
        if self.algorithm not in ("telea", "navier-stokes"):
            raise ValueError("Unsupported glyph inpainting algorithm")
        if type(self.fill_enclosed_white) is not bool:
            raise ValueError("Enclosed-white selection must be boolean")
        if not 1 <= self.expansion <= 6:
            raise ValueError("Glyph expansion must be between 1 and 6 pixels")
        if not isfinite(self.softness) or not 0 <= self.softness <= 1:
            raise ValueError("Softness must be finite and normalized")


def validate_roi(image: Image) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 BGR image")
    if min(image.shape[:2]) < 8 or image.shape[0] * image.shape[1] > 2_000_000:
        raise ValueError("Invalid or excessive subtitle ROI")


def unchanged_outside_support(source: Image, result: Image, support: Image) -> bool:
    """Exact byte comparison without gathering copies of every unmasked pixel."""
    validate_roi(source)
    validate_roi(result)
    if (
        source.shape != result.shape
        or support.shape != source.shape[:2]
        or support.dtype != np.uint8
    ):
        raise ValueError("Invalid support comparison shapes or dtype")
    return cv2.norm(source, result, cv2.NORM_INF, np.asarray(support == 0, dtype=np.uint8)) == 0


def pink_glyph_mask(
    image: Image, *, expansion: int = 2, fill_enclosed_white: bool = False
) -> Image:
    """Profile-specific pink seeds with optional enclosed neutral-white interiors.

    Only pass an independently localized subtitle ROI. This color profile is
    not a universal subtitle detector and must not scan unrelated image areas.
    The opt-in contour fill needs independent real-output acceptance; it neither
    removes non-text pink seeds nor reconstructs hidden background structure.
    """
    validate_roi(image)
    InpaintSettings(expansion=expansion, fill_enclosed_white=fill_enclosed_white)
    colors = image.astype(np.int16)
    seeds = (colors[:, :, 2] - colors[:, :, 1] > 24) & (colors[:, :, 0] - colors[:, :, 1] > 8)
    mask = seeds.astype(np.uint8) * 255
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    if fill_enclosed_white:
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enclosed = np.zeros_like(mask)
        cv2.drawContours(enclosed, contours, -1, 255, cv2.FILLED)
        white = (np.min(colors, axis=2) >= 192) & (np.ptp(colors, axis=2) <= 32)
        closed[white & (enclosed != 0)] = 255
    return np.asarray(
        cv2.dilate(closed, np.ones((3, 3), dtype=np.uint8), iterations=expansion),
        dtype=np.uint8,
    )


def repair_glyphs(
    image: Image, *, settings: InpaintSettings, text_mask: Image | None = None
) -> tuple[Image, Image]:
    """Use local glyph inpainting; default retries retain Telea and local support.

    Returns the repaired BGR ROI and allowed modification support. Renderer
    callers must ramp softness in time; this spatial kernel does not establish
    temporal consistency or visual acceptance on its own.
    """
    validate_roi(image)
    if text_mask is None:
        # Retained only for explicit legacy diagnostics; the video path supplies Vision masks.
        mask = pink_glyph_mask(
            image, expansion=settings.expansion, fill_enclosed_white=settings.fill_enclosed_white
        )
    else:
        if text_mask.dtype != np.uint8 or text_mask.shape != image.shape[:2]:
            raise ValueError("Invalid character mask shape or dtype")
        if not np.all((text_mask == 0) | (text_mask == 255)):
            raise ValueError("Character mask must be binary")
        mask = np.asarray(
            cv2.dilate(text_mask, np.ones((3, 3), dtype=np.uint8), iterations=settings.expansion),
            dtype=np.uint8,
        )
    if not np.any(mask):
        return image.copy(), mask
    algorithm = cv2.INPAINT_NS if settings.algorithm == "navier-stokes" else cv2.INPAINT_TELEA
    repaired = np.asarray(cv2.inpaint(image, mask, 3, algorithm), dtype=np.uint8)
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
    cv2.copyTo(image, np.asarray(support == 0, dtype=np.uint8), result)
    return result, support


def registered_glyph_samples(
    source: Image,
    donor: Image,
    *,
    source_mask: Image,
    donor_mask: Image,
    forward: Flow,
    backward: Flow,
) -> tuple[Image, Image]:
    """Copy only supported donor samples through independently estimated ROI flows.

    Forward maps source pixels to donor coordinates; backward maps donor to source.
    Reject out-of-bounds footprints, any donor-mask contamination and round-trip
    error over half a pixel. Unavailable pixels stay unchanged, not inpainted.
    Consistent flows alone do not prove correspondence, absence of subtitles or
    temporal quality. Callers must independently establish those before delivery.
    """
    validate_roi(source)
    validate_roi(donor)
    height, width = source.shape[:2]
    if donor.shape != source.shape:
        raise ValueError("Registered source and donor shapes differ")
    if any(
        mask.shape != (height, width) or mask.dtype != np.uint8
        for mask in (source_mask, donor_mask)
    ):
        raise ValueError("Invalid registered glyph masks")
    if any(
        flow.shape != (height, width, 2)
        or flow.dtype != np.float32
        or not np.isfinite(flow).all()
        or np.max(np.abs(flow)) > max(height, width)
        for flow in (forward, backward)
    ):
        raise ValueError("Invalid registered pixel flows")
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    map_x, map_y = grid_x + forward[:, :, 0], grid_y + forward[:, :, 1]
    inside = (map_x >= 0) & (map_y >= 0) & (map_x < width - 1) & (map_y < height - 1)
    sampled_back = cv2.remap(backward, map_x, map_y, cv2.INTER_LINEAR)
    cycle = forward + sampled_back
    consistent = np.hypot(cycle[:, :, 0], cycle[:, :, 1]) <= MAX_ROUND_TRIP_ERROR
    contaminated = cv2.remap(
        np.asarray(donor_mask != 0, dtype=np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1,
    )
    accepted = (
        np.asarray((source_mask != 0) & inside & consistent & (contaminated == 0), dtype=np.uint8)
        * 255
    )
    warped = cv2.remap(donor, map_x, map_y, cv2.INTER_LINEAR)
    result = source.copy()
    cv2.copyTo(warped, accepted, result)
    return result, accepted


def ramp_softness(
    *,
    frame: int,
    frames: int,
    target: float,
    ramp_frames: int = 6,
    start_frame: int = 0,
    total_frames: int | None = None,
) -> float:
    """Smoothstep at sequence boundaries, not arbitrary processing-chunk seams."""
    total = frames if total_frames is None else total_frames
    if (
        any(type(value) is not int for value in (frame, frames, ramp_frames, start_frame, total))
        or frames < 1
        or not 0 <= frame < frames
        or ramp_frames < 1
        or start_frame < 0
        or start_frame + frames > total
    ):
        raise ValueError("Invalid softness ramp clocks")
    if not isfinite(target) or not 0 <= target <= 1:
        raise ValueError("Invalid target softness")
    position = start_frame + frame
    distance = min(position, total - 1 - position)
    fraction = min(1.0, distance / ramp_frames)
    return target * fraction * fraction * (3 - 2 * fraction)
