"""Explicit, hash-bound local PP-OCRv3 DB detection; never OCR-derived masks."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .local_asr import fingerprint
from .subtitle_inpaint import Image, validate_roi

MAX_MODEL_BYTES = 32 * 1024 * 1024
DETECTOR_SIZE = 736
POLYGON_SHIFT = 8


def polygon_mask(
    polygons: object,
    confidences: object,
    *,
    width: int,
    height: int,
    grid: tuple[int, int] = (DETECTOR_SIZE, DETECTOR_SIZE),
) -> Image:
    """Project validated detector-grid polygons onto the original pixel canvas.

    DB's configured unclip expansion may cross the model canvas. Intersect that
    measured convex polygon with the canvas, never clamp individual vertices or
    add support. Reject excessive expansion, invalid or wholly off-canvas geometry.
    Fixed-point projection retains subpixel corners on small source ROIs.
    """
    if min(width, height) < 8 or width * height > 2_000_000:
        raise ValueError("Invalid detector canvas")
    if len(grid) != 2 or any(
        type(value) is not int or not 32 <= value <= DETECTOR_SIZE or value % 32 for value in grid
    ):
        raise ValueError("Invalid model grid")
    bounds = np.array(grid, dtype=np.int32)
    if not isinstance(polygons, (list, tuple)) or len(polygons) > 64:
        raise ValueError("Invalid or excessive detected polygons")
    if (
        not isinstance(confidences, np.ndarray)
        or confidences.dtype.kind != "f"
        or confidences.ndim not in (1, 2)
        or confidences.size != len(polygons)
        or not np.isfinite(confidences).all()
        or np.any((confidences < 0.5) | (confidences > 1))
    ):
        raise ValueError("Invalid detector confidence evidence")
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if (
            not isinstance(polygon, np.ndarray)
            or polygon.dtype != np.int32
            or polygon.shape != (4, 2)
            or np.any(polygon < -bounds)
            or np.any(polygon > 2 * bounds)
            or not cv2.isContourConvex(polygon)
            or cv2.contourArea(polygon) <= 0
        ):
            raise ValueError("Invalid measured text polygon")
        area, intersection = cv2.intersectConvexConvex(
            polygon.astype(np.float32),
            np.array(
                [[0, 0], [grid[0], 0], [grid[0], grid[1]], [0, grid[1]]],
                dtype=np.float32,
            ),
        )
        if area <= 0 or intersection is None:
            raise ValueError("Detected polygon has no canvas support")
        scale = np.array([width, height], dtype=np.float64) / bounds
        projected = np.rint(intersection.reshape(-1, 2) * scale * (1 << POLYGON_SHIFT)).astype(
            np.int32
        )
        cv2.fillPoly(mask, [projected], 255, shift=POLYGON_SHIFT)
    return mask


class LearnedTextDetector:
    """One worker owns one CPU detector. Loading caps OpenCV threads at two.

    Uses the official OpenCV Zoo FP32 PP-OCRv3 DB preprocessing/decoding policy.
    No download, executable model code, OCR rectangles or automatic fallback.
    A detection-only copy is resized to the required model grid. Original source
    pixels are untouched; measured polygons are projected onto the ROI grid.
    Empty output is not absence; inference does not certify coverage or quality.
    """

    def __init__(self, model: Path, *, sha256: str) -> None:
        if (
            model.is_symlink()
            or not model.is_file()
            or not 0 < model.stat().st_size <= MAX_MODEL_BYTES
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or fingerprint(model) != sha256
        ):
            raise ValueError("Expected a bounded, hash-verified local detector model")
        self.grid_sizes: list[tuple[int, int]] = []
        self.square_context_regions: list[int] = []
        self.observed_regions: list[int] = []
        self.intersected_regions: list[int] = []
        cv2.setNumThreads(2)
        self._model = cv2.dnn.TextDetectionModel_DB(cv2.dnn.readNet(str(model)))
        self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._model.setBinaryThreshold(0.3)
        self._model.setPolygonThreshold(0.5)
        self._model.setMaxCandidates(200)
        self._model.setUnclipRatio(2.0)
        self._model.setInputSize((DETECTOR_SIZE, DETECTOR_SIZE))
        self._model.setInputMean((123.675, 116.28, 103.53))
        self._model.setInputScale(tuple(1.0 / 255.0 / value for value in (0.229, 0.224, 0.225)))
        if fingerprint(model) != sha256:
            raise ValueError("Detector model changed during loading")

    def _view(self, image: Image, grid: tuple[int, int]) -> tuple[Image, int, int]:
        self._model.setInputSize(grid)
        polygons, confidences = self._model.detect(cv2.resize(image, grid))
        mask = polygon_mask(
            polygons,
            np.asarray(confidences, dtype=np.float64),
            width=image.shape[1],
            height=image.shape[0],
            grid=grid,
        )
        intersections = sum(
            1
            for polygon in polygons
            if np.any(np.asarray(polygon, dtype=np.int32) < 0)
            or np.any(np.asarray(polygon, dtype=np.int32) > np.array(grid, dtype=np.int32))
        )
        return mask, len(polygons), intersections

    def detect(self, images: Sequence[Image]) -> tuple[Image, ...]:
        """Bound sequential inference; no whole-video pixel retention or mask reuse."""
        if not 1 <= len(images) <= 8:
            raise ValueError("Supply 1–8 detector ROIs")
        masks: list[Image] = []
        for image in images:
            validate_roi(image)
            height, width = image.shape[:2]
            scale = DETECTOR_SIZE / max(width, height)
            grid = (
                max(32, round(width * scale / 32) * 32),
                max(32, round(height * scale / 32) * 32),
            )
            mask, regions, intersections = self._view(image, grid)
            square_regions = 0
            if grid != (DETECTOR_SIZE, DETECTOR_SIZE):
                # Complementary measured views, not an absence-triggered fallback.
                square, square_regions, square_intersections = self._view(
                    image, (DETECTOR_SIZE, DETECTOR_SIZE)
                )
                if regions + square_regions > 64:
                    raise ValueError("Combined detector region budget exceeded")
                mask = np.maximum(mask, square)
                regions += square_regions
                intersections += square_intersections
            masks.append(mask)
            self.grid_sizes.append(grid)
            self.square_context_regions.append(square_regions)
            self.observed_regions.append(regions)
            self.intersected_regions.append(intersections)
        return tuple(masks)
