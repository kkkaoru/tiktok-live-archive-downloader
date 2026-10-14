"""Per-line support observations and explicitly reviewed, bounded ink repair.

DB polygons and OCR envelopes locate candidates, not ink. Chroma proposals below
are deliberately incomplete and never auto-approved. Callers must inspect and bind
source/line/mask evidence before supplying reviewed ink to the repair function.
"""

from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np

from .caption_masks import TextDetection, TextRectangle
from .subtitle_inpaint import (
    Image,
    InpaintSettings,
    repair_glyphs,
    unchanged_outside_support,
    validate_roi,
)


@dataclass(frozen=True, slots=True)
class ComponentAttribution:
    component: int
    pixels: int
    box: tuple[int, int, int, int]
    overlapping_lines: tuple[int, ...]
    selected_line: int | None
    outside_pixels: int


def line_masks(lines: tuple[TextDetection, ...], *, region: TextRectangle) -> tuple[Image, ...]:
    """Rasterize original OCR envelopes separately, with no padding or unions."""
    if not 1 <= len(lines) <= 64 or any(
        value != int(value) for value in (region.x, region.y, region.width, region.height)
    ):
        raise ValueError("Expected bounded lines and integer ROI")
    width, height = int(region.width), int(region.height)
    if min(width, height) < 8 or width * height > 2_000_000:
        raise ValueError("Invalid line attribution canvas")
    masks: list[Image] = []
    for line in lines:
        box = line.rectangle
        left, top = max(0, floor(box.x - region.x)), max(0, floor(box.y - region.y))
        right = min(width, ceil(box.x + box.width - region.x))
        bottom = min(height, ceil(box.y + box.height - region.y))
        mask = np.zeros((height, width), dtype=np.uint8)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 255
        masks.append(mask)
    return tuple(masks)


def _validate_masks(
    support: Image, envelopes: tuple[Image, ...], selected: tuple[int, ...]
) -> None:
    if (
        support.dtype != np.uint8
        or support.ndim != 2
        or min(support.shape) < 8
        or support.size > 2_000_000
        or not np.all((support == 0) | (support == 255))
        or not 1 <= len(envelopes) <= 64
        or not selected
        or selected != tuple(sorted(set(selected)))
        or not 0 <= selected[0] <= selected[-1] < len(envelopes)
    ):
        raise ValueError("Invalid support or selected line indices")
    if any(
        mask.dtype != np.uint8
        or mask.shape != support.shape
        or not np.all((mask == 0) | (mask == 255))
        for mask in envelopes
    ):
        raise ValueError("Invalid line envelopes")


def attribute_components(
    support: Image,
    *,
    envelopes: tuple[Image, ...],
    selected: tuple[int, ...],
    localization_only: bool = False,
) -> tuple[ComponentAttribution, ...]:
    """Attribute only whole components inside exactly one selected original line.

    Crossing components remain unknown rather than clipped into plausible glyphs.
    With explicit localization_only, outside pixels are reported, not rejected:
    callers must independently review permission beyond the OCR envelope. Multiple
    line overlaps and components without an OCR anchor still remain unknown.
    An attributed component is still NOT certified caption ink.
    """
    _validate_masks(support, envelopes, selected)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    if count > 4097:
        raise ValueError("Excessive support components")
    union = np.maximum.reduce([envelopes[index] for index in selected])
    result: list[ComponentAttribution] = []
    for component in range(1, count):
        pixels = labels == component
        overlaps = tuple(
            index for index, mask in enumerate(envelopes) if np.any(pixels & (mask != 0))
        )
        outside = int(np.count_nonzero(pixels & (union == 0)))
        owner = (
            overlaps[0]
            if len(overlaps) == 1
            and overlaps[0] in selected
            and (outside == 0 or localization_only)
            else None
        )
        x, y, width, height, area = (int(value) for value in stats[component])
        result.append(
            ComponentAttribution(component, area, (x, y, width, height), overlaps, owner, outside)
        )
    return tuple(result)


def chroma_proposals(
    image: Image, *, regions: Image, envelopes: tuple[Image, ...], selected: tuple[int, ...]
) -> tuple[tuple[Image, ...], tuple[float, ...]]:
    """Observe high-chroma cores using per-line Otsu; not a general ink classifier.

    No hue, recording ID, padding or automatic white/interior fill. Monochrome
    captions can yield no useful proposal. Colourful decoration may also be
    selected: explicit pixel review is mandatory, not inferred from OCR or DB.
    """
    validate_roi(image)
    _validate_masks(regions, envelopes, selected)
    if regions.shape != image.shape[:2]:
        raise ValueError("Source and region shapes differ")
    chroma = np.ptp(image, axis=2)
    proposals: list[Image] = []
    thresholds: list[float] = []
    for index in selected:
        domain = (regions != 0) & (envelopes[index] != 0)
        values = chroma[domain]
        if values.size == 0 or np.min(values) == np.max(values):
            threshold = float(np.max(values)) if values.size else 0.0
            proposal = np.zeros_like(regions)
        else:
            threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            proposal = np.asarray(domain & (chroma > threshold), dtype=np.uint8) * 255
        proposals.append(proposal)
        thresholds.append(threshold)
    return tuple(proposals), tuple(thresholds)


def connected_chroma_proposal(
    image: Image,
    *,
    regions: Image,
    seed: Image,
    threshold: float,
    envelopes: tuple[Image, ...],
    selected: tuple[int, ...],
) -> Image:
    """Observe same-chroma components beyond an OCR box, without granting ink.

    A component must connect to the supplied seed and overlap exactly one selected
    line. Background of similar colour may connect too: retain for source review,
    never silently label the extension or its repair halo as approved text.
    """
    validate_roi(image)
    _validate_masks(regions, envelopes, selected)
    _validate_masks(seed, envelopes, selected)
    if (
        image.shape[:2] != regions.shape
        or seed.shape != regions.shape
        or not 0 <= threshold <= 255
        or np.any((seed != 0) & (regions == 0))
    ):
        raise ValueError("Invalid connected chroma evidence")
    proposed = (
        np.asarray((regions != 0) & (np.ptp(image, axis=2) > threshold), dtype=np.uint8) * 255
    )
    attributed = attribute_components(
        proposed, envelopes=envelopes, selected=selected, localization_only=True
    )
    _, labels = cv2.connectedComponents(proposed, connectivity=8)
    retained = np.zeros_like(seed)
    for row in attributed:
        component = labels == row.component
        if row.selected_line is not None and np.any(component & (seed != 0)):
            retained[component] = 255
    return retained


def bounded_proposals(
    ink: Image, *, envelopes: tuple[Image, ...], selected: tuple[int, ...]
) -> tuple[Image, Image]:
    """Keep whole seed components only when their GLYPH halo has one line owner.

    Returned rejected seeds stay unresolved; this is a boundary filter, not ink
    classification. Never trim a crossing component to manufacture eligibility.
    """
    rows = attribute_components(ink, envelopes=envelopes, selected=selected)
    _, labels = cv2.connectedComponents(ink, connectivity=8)
    retained = np.zeros_like(ink)
    for row in rows:
        component = np.asarray(labels == row.component, dtype=np.uint8) * 255
        expanded = np.asarray(
            cv2.dilate(component, np.ones((3, 3), dtype=np.uint8), iterations=2), dtype=np.uint8
        )
        owners = attribute_components(expanded, envelopes=envelopes, selected=selected)
        if row.selected_line is not None and all(
            item.selected_line == row.selected_line for item in owners
        ):
            retained |= component
    return retained, np.asarray(cv2.subtract(ink, retained), dtype=np.uint8)


def reviewed_support(
    image: Image,
    *,
    ink: Image,
    allowed: Image,
    envelopes: tuple[Image, ...],
    selected: tuple[int, ...],
    localization_only: bool = False,
) -> Image:
    """Validate reviewed ink and its permission before running any repair.

    Masks must be independently reviewed, source-bound evidence, not detector
    polygons or OCR boxes. Dilation does not grant permission.
    """
    validate_roi(image)
    _validate_masks(ink, envelopes, selected)
    _validate_masks(allowed, envelopes, selected)
    if ink.shape != image.shape[:2] or not np.any(ink) or np.any((ink != 0) & (allowed == 0)):
        raise ValueError("Missing reviewed ink or invalid source support")
    # OCR anchors the ink, not its halo. Reviewed halos may cross an inaccurate
    # neighbouring OCR box, but cannot bridge independently reviewed line masks.
    attribution_mask = ink if localization_only else allowed
    if any(
        row.selected_line is None
        for row in attribute_components(
            attribution_mask,
            envelopes=envelopes,
            selected=selected,
            localization_only=localization_only,
        )
    ):
        raise ValueError("Unknown, crossing or unselected repair component")
    expected = np.asarray(
        cv2.dilate(ink, np.ones((3, 3), dtype=np.uint8), iterations=2), dtype=np.uint8
    )
    if localization_only and np.any((allowed != 0) & (expected == 0)):
        raise ValueError("Unknown permission beyond reviewed ink halo")
    if np.any((expected != 0) & (allowed == 0)):
        raise ValueError("Glyph expansion exceeds reviewed permission")
    return expected


def repair_reviewed_ink(
    image: Image,
    *,
    ink: Image,
    allowed: Image,
    envelopes: tuple[Image, ...],
    selected: tuple[int, ...],
    localization_only: bool = False,
) -> tuple[Image, Image]:
    """Run the existing GLYPH kernel once; never fabricate quality acceptance."""
    expected = reviewed_support(
        image,
        ink=ink,
        allowed=allowed,
        envelopes=envelopes,
        selected=selected,
        localization_only=localization_only,
    )
    candidate, actual = repair_glyphs(image, settings=InpaintSettings(expansion=2), text_mask=ink)
    if not np.array_equal(actual, expected) or not unchanged_outside_support(
        image, candidate, allowed
    ):
        raise ValueError("Repair escaped reviewed support")
    return candidate, actual
