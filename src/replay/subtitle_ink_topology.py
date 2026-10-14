"""Pixel observations for bright fill and light rims; never automatic permission.

Supported bright components require a darker measured exterior and core evidence.
An enclosed white counter on a white background is ambiguous and stays unknown.
An outlined decoration can be indistinguishable from a glyph: independent source
review is mandatory. No contour is painted wholesale into the returned mask.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .subtitle_inpaint import Image, validate_roi


@dataclass(frozen=True, slots=True)
class CavityObservation:
    contour: int
    pixels: int
    proposed_pixels: int
    inside_median: float
    exterior_median: float | None
    threshold: float | None


def bright_component_proposals(image: Image, *, core: Image, region: Image) -> Image:
    """Observe open-rim fill and bright outside strokes, without filling holes.

    Brightness Otsu is measured in the localized region but applied to the WHOLE
    image: crossing white background components cannot be cropped into glyphs.
    Fill needs a majority-core boundary. A thin outside rim instead needs every
    pixel within one measured core diameter and actual core contact. Both require
    a darker exterior outside the combined rim. A region is localization, not a
    pixel fence: complete components are retained or rejected, never clipped.
    If enclosed bright fill is observed, an outside bright rim is not inferred:
    touching scene highlights and mixed fill/rim styles remain unresolved. White-
    background counters remain unknown. This is not semantic approval.
    """
    validate_roi(image)
    if any(
        mask.dtype != np.uint8
        or mask.shape != image.shape[:2]
        or mask.size > 2_000_000
        or not np.all((mask == 0) | (mask == 255))
        for mask in (core, region)
    ):
        raise ValueError("Invalid bright component evidence")
    result = np.zeros_like(core)
    brightness = np.min(image, axis=2)
    values = brightness[region != 0]
    if values.size == 0 or np.min(values) == np.max(values):
        return result
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = np.asarray(brightness > threshold, dtype=np.uint8) * 255
    count, labels = cv2.connectedComponents(bright, connectivity=8)
    if count > 4097:
        raise ValueError("Excessive bright components")
    kernel = np.ones((3, 3), dtype=np.uint8)
    enclosed_fill, _ = bright_fill_proposals(image, core=core)
    has_bright_fill = bool(np.any(enclosed_fill))
    diameter = 2 * float(np.max(cv2.distanceTransform(core, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)))
    distance = cv2.distanceTransform(255 - core, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    for index in range(1, count):
        component = labels == index
        if not np.any(component & (region != 0)):
            continue
        mask = np.asarray(component, dtype=np.uint8) * 255
        boundary = (cv2.dilate(mask, kernel) != 0) & ~component
        contact = int(np.count_nonzero(boundary & (core != 0)))
        fill = contact * 2 > np.count_nonzero(boundary) and not np.any(component & (region == 0))
        thin_rim = (
            not has_bright_fill and contact > 0 and float(np.max(distance[component])) <= diameter
        )
        if not fill and not thin_rim:
            continue
        # Measure outside the entire connected rim, not across its own white fill.
        combined = core | mask
        _, combined_labels = cv2.connectedComponents(combined, connectivity=8)
        owner = int(combined_labels[component][0])
        rim = np.asarray(combined_labels == owner, dtype=np.uint8) * 255
        contours, _ = cv2.findContours(rim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outside = np.zeros_like(core)
        cv2.drawContours(outside, contours, -1, 255, cv2.FILLED)
        exterior = (cv2.dilate(outside, kernel) != 0) & (outside == 0) & (bright == 0)
        # Bright exterior cannot be discarded to manufacture dark-background proof.
        full_exterior = (cv2.dilate(outside, kernel) != 0) & (outside == 0)
        if np.any(exterior) and float(np.median(brightness[component])) > float(
            np.median(brightness[full_exterior])
        ):
            result[component & (core == 0)] = 255
    return result


def bright_fill_proposals(
    image: Image, *, core: Image
) -> tuple[Image, tuple[CavityObservation, ...]]:
    """Observe bright pixels, comparing each rim cavity with its own exterior.

    The supplied chroma core must itself be independently inspected; no hue is
    assumed. RETR_TREE retains nested topology. A cavity's exterior is measured
    outside its parent rim, not in its rim or a sibling cavity. Child islands and
    all non-bright pixels remain excluded. Missing/indistinguishable exterior
    provides no proposal. Results are incomplete observations, not an ink label.
    """
    validate_roi(image)
    if (
        core.dtype != np.uint8
        or core.shape != image.shape[:2]
        or core.size > 2_000_000
        or not np.all((core == 0) | (core == 255))
    ):
        raise ValueError("Invalid outline core")
    contours, hierarchy = cv2.findContours(core, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    proposal = np.zeros_like(core)
    if hierarchy is None:
        return proposal, ()
    if len(contours) > 4096:
        raise ValueError("Excessive outline contours")
    brightness = np.min(image, axis=2)
    observations: list[CavityObservation] = []
    depths: list[int] = []
    for index, contour in enumerate(contours):
        parent = int(hierarchy[0, index, 3])
        depth = 0 if parent < 0 else depths[parent] + 1
        depths.append(depth)
        if depth % 2 == 0:
            continue
        cavity = np.zeros_like(core)
        cv2.drawContours(cavity, [contour], -1, 255, cv2.FILLED)
        child = int(hierarchy[0, index, 2])
        while child >= 0:
            cv2.drawContours(cavity, contours, child, 0, cv2.FILLED)
            child = int(hierarchy[0, child, 0])
        interior = (cavity != 0) & (core == 0)
        if not np.any(interior):
            continue
        parent_mask = np.zeros_like(core)
        cv2.drawContours(parent_mask, contours, parent, 255, cv2.FILLED)
        exterior = (
            (cv2.dilate(parent_mask, np.ones((3, 3), dtype=np.uint8)) != 0)
            & (parent_mask == 0)
            & (core == 0)
        )
        inside_median = float(np.median(brightness[interior]))
        outside_median = float(np.median(brightness[exterior])) if np.any(exterior) else None
        threshold: float | None = None
        selected = np.zeros_like(core, dtype=np.bool_)
        if outside_median is not None and inside_median > outside_median:
            values = np.concatenate((brightness[interior], brightness[exterior]))
            threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            selected = interior & (brightness > threshold)
            proposal[selected] = 255
        observations.append(
            CavityObservation(
                index,
                int(np.count_nonzero(interior)),
                int(np.count_nonzero(selected)),
                inside_median,
                outside_median,
                threshold,
            )
        )
    return proposal, tuple(observations)
