"""Explicit local-affine context geometry, not hidden-background correspondence."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

Coordinates = NDArray[np.float32]
Support = NDArray[np.bool_]
MIN_PARTITION = 8
NEIGHBORS = 2 * MIN_PARTITION
MAX_DISTANCE_PIXELS = 128  # Four times the original track exclusion/window margin.
ERROR_PIXELS = 0.5
INLIER_FRACTION = 0.9


@dataclass
class GeometryDiagnostics:
    """Exclusive neighborhood rejection counts; residual partitions can overlap."""

    valid_tracks: int
    invalid_tracks: int
    insufficient_neighbors: int = 0
    distant_neighbors: int = 0
    invalid_neighbors: int = 0
    rank: int = 0
    residual: int = 0
    training_residual: int = 0
    held_residual: int = 0
    hull: int = 0
    agreement_pixels: int = 0


@dataclass(frozen=True)
class LocalGeometry:
    donor_xy: Coordinates
    supported: Support
    tested: int
    accepted: int
    diagnostics: GeometryDiagnostics


def local_affine_geometry(
    points: Coordinates,
    *,
    targets: Coordinates,
    valid: Support,
    shape: tuple[int, int],
) -> LocalGeometry:
    """Fit bounded neighborhoods using original-only bidirectionally checked tracks.

    Each neighborhood needs eight fitting and eight independent held-out tracks.
    Both partitions must meet the unchanged 90% / half-pixel error policy. No
    RANSAC cherry-picking or candidate pixels participate. Only the intersection
    of the two partitions' convex hulls supports interpolation; no extrapolation.
    The coordinate-range diagonal of overlapping models must be <=0.5px (a
    conservative bound on every pair's disagreement), never an averaged pass.
    Exclusion, donor footprint, photometry and occlusion remain caller checks.
    Even supported geometry cannot observe motion hidden behind source text.
    """
    height, width = shape
    if (
        min(shape) < 8
        or height * width > 2_000_000
        or points.dtype != np.float32
        or targets.dtype != np.float32
        or points.ndim != 2
        or points.shape[1] != 2
        or targets.shape != points.shape
        or not 0 <= len(points) <= 256
        or valid.dtype != np.bool_
        or valid.shape != (len(points),)
        or not np.isfinite(points).all()
        or not np.isfinite(targets[valid]).all()
        or np.any(points < 0)
        or np.any(points >= (width, height))
    ):
        raise ValueError("Invalid local affine track geometry")
    yy, xx = np.indices(shape, dtype=np.float32)
    grid = np.stack((xx, yy), axis=-1)
    mapped = grid.copy()
    supported = np.zeros(shape, np.bool_)
    lower = np.full_like(grid, np.inf)
    upper = np.full_like(grid, -np.inf)
    tested = accepted = 0
    diagnostics = GeometryDiagnostics(int(np.count_nonzero(valid)), int(np.count_nonzero(~valid)))
    for point in points:
        distance = np.linalg.norm(points - point, axis=1)
        neighbors = np.argsort(distance, kind="stable")[:NEIGHBORS]
        if len(neighbors) < NEIGHBORS:
            diagnostics.insufficient_neighbors += 1
            continue
        if distance[neighbors[-1]] > MAX_DISTANCE_PIXELS:
            diagnostics.distant_neighbors += 1
            continue
        tested += 1
        if not np.all(valid[neighbors]):
            diagnostics.invalid_neighbors += 1
            continue
        training, held = neighbors[::2], neighbors[1::2]
        design = np.asarray(
            np.column_stack((points[training], np.ones(MIN_PARTITION))), dtype=np.float64
        )
        matrix, _, rank, _ = np.linalg.lstsq(design, targets[training], rcond=None)
        if rank != 3:
            diagnostics.rank += 1
            continue
        validation = np.asarray(
            np.column_stack((points[neighbors], np.ones(NEIGHBORS))), dtype=np.float64
        )
        predicted = validation @ matrix
        error = np.linalg.norm(predicted - targets[neighbors], axis=1)
        training_failed = bool(np.mean(error[::2] <= ERROR_PIXELS) < INLIER_FRACTION)
        held_failed = bool(np.mean(error[1::2] <= ERROR_PIXELS) < INLIER_FRACTION)
        if training_failed or held_failed:
            diagnostics.residual += 1
            diagnostics.training_residual += int(training_failed)
            diagnostics.held_residual += int(held_failed)
            continue
        # Do not expand floating hulls by rounding corners outward.
        hull_train = cv2.convexHull(points[training])
        hull_held = cv2.convexHull(points[held])
        area, hull = cv2.intersectConvexConvex(hull_train, hull_held)
        if area <= 0 or hull is None:
            diagnostics.hull += 1
            continue
        left, top, box_width, box_height = cv2.boundingRect(hull)
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, left + box_width), min(height, top + box_height)
        local = grid[top:bottom, left:right]
        inside = _inside_hull(local, np.asarray(hull.reshape(-1, 2), dtype=np.float32))
        proposal = np.asarray(local @ matrix[:2] + matrix[2], np.float32)
        existing = mapped[top:bottom, left:right]
        support = supported[top:bottom, left:right]
        low, high = lower[top:bottom, left:right], upper[top:bottom, left:right]
        low[inside] = np.minimum(low[inside], proposal[inside])
        high[inside] = np.maximum(high[inside], proposal[inside])
        existing[inside & ~support] = proposal[inside & ~support]
        support |= inside
        accepted += 1
    agreement = np.linalg.norm(upper - lower, axis=2) <= ERROR_PIXELS
    diagnostics.agreement_pixels = int(np.count_nonzero(supported & ~agreement))
    supported &= agreement
    return LocalGeometry(mapped, supported, tested, accepted, diagnostics)


def _inside_hull(grid: Coordinates, vertices: Coordinates) -> Support:
    inside = np.ones(grid.shape[:2], np.bool_)
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        edge = end - start
        inside &= edge[0] * (grid[:, :, 1] - start[1]) - edge[1] * (grid[:, :, 0] - start[0]) >= 0
    return inside
