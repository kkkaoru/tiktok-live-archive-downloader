"""Derive contour motion from measured foreground texture, never a new detection."""

from dataclasses import dataclass
from itertools import chain, zip_longest

import cv2
import numpy as np

from replay.head_seed import ContourTrackMetrics, Coordinates, measure_contour_tracks
from replay.subtitle_inpaint import Image

Matrix = tuple[tuple[float, float, float], tuple[float, float, float]]
Region = tuple[int, int, int, int]
Point = tuple[float, float]
MAX_TEXTURE_FEATURES = 256
FEATURE_SPACING_PIXELS = 7
LOCAL_SUPPORT_POLICY = "predeclared-measured-region-spatial-v1"


@dataclass(frozen=True, slots=True)
class SpatialFeatureSelection:
    """Current independent points and the spatial/temporal selection evidence."""

    cohorts: tuple[Coordinates, Coordinates]
    occupied_pixels: tuple[int, ...]
    continued_candidates: tuple[int, ...]
    fresh_candidates: tuple[int, ...]
    selected_continued: tuple[int, ...]
    selected_fresh: tuple[int, ...]
    origins: tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]
    split: tuple[int, int]
    rejected_continued: int

    def audit(self) -> dict[str, object]:
        return {
            "policy": "current-supported-spatial-replenishment-v1",
            "areaMedianSplit": list(self.split),
            "cellEligiblePixels": list(self.occupied_pixels),
            "cellContinuedCandidates": list(self.continued_candidates),
            "cellFreshCandidates": list(self.fresh_candidates),
            "cellSelectedContinued": list(self.selected_continued),
            "cellSelectedFresh": list(self.selected_fresh),
            "selectedOrigins": [list(group) for group in self.origins],
            "rejectedContinuedOutsideCurrentSupport": self.rejected_continued,
            "selectedFeatureCounts": [len(group) for group in self.cohorts],
            "selectedSpans": [
                np.ptp(group, axis=0).tolist() if len(group) else [0.0, 0.0]
                for group in self.cohorts
            ],
            "selectionBeforeNextFrame": True,
            "nextFrameRevalidatesEverySelectedPoint": True,
            "newFaceObservation": False,
        }


@dataclass(frozen=True, slots=True)
class LocalFeatureSelection:
    """Reassign current selected points to measured owners before their next fit."""

    cohorts: tuple[Coordinates, Coordinates]
    origins: tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]
    original_selection: SpatialFeatureSelection
    source_indices: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    budget_excluded: tuple[tuple[int, int], ...]
    tile_owners: tuple[int, ...]
    seed_split_y: int

    def audit(self) -> dict[str, object]:
        result = self.original_selection.audit()
        result.update(
            {
                "selectedFeatureCounts": [len(group) for group in self.cohorts],
                "selectedOrigins": self.origins,
                "currentSelectedFeatures": [group.tolist() for group in self.cohorts],
                "selectedSpans": [
                    np.ptp(group, axis=0).tolist() if len(group) else [0.0, 0.0]
                    for group in self.cohorts
                ],
                "cellSelectedContinued": _local_cell_counts(self, fresh=False),
                "cellSelectedFresh": _local_cell_counts(self, fresh=True),
                "predeclaredLocalSupport": {
                    "policy": LOCAL_SUPPORT_POLICY,
                    "seedAreaMedianY": self.seed_split_y,
                    "originalTileOwners": self.tile_owners,
                    "originalSelectedIndices": self.source_indices,
                    "ownerBudgetExcludedIndices": self.budget_excluded,
                    "maximumFeaturesPerOwner": MAX_TEXTURE_FEATURES // 2,
                    "ownershipSelectedBeforeFit": True,
                    "fitDependentSelection": False,
                    "originalSelection": self.original_selection.audit(),
                    "originalSelectedFeatures": [
                        group.tolist() for group in self.original_selection.cohorts
                    ],
                },
            }
        )
        return result


def _local_cell_counts(selection: LocalFeatureSelection, *, fresh: bool) -> tuple[int, ...]:
    material = tuple(
        chain.from_iterable(
            zip(group, origins, strict=True)
            for group, origins in zip(selection.cohorts, selection.origins, strict=True)
        )
    )
    return tuple(
        sum(
            _cell((float(point[0]), float(point[1])), selection.original_selection.split) == cell
            and (origin[0] == "fresh") is fresh
            for point, origin in material
        )
        for cell in range(4)
    )


def measured_region_owners(
    image: Image, regions: tuple[Region, ...]
) -> tuple[int, tuple[int, ...]]:
    """Fix whole-tile model ownership from original occupied area, without a fit."""
    _region_mask(image, regions)
    return measured_tile_partition(regions, width=image.shape[1], height=image.shape[0])


def measured_tile_partition(
    regions: tuple[Region, ...], *, width: int, height: int
) -> tuple[int, tuple[int, ...]]:
    """Recompute measured ownership from bounded geometry without decoding pixels."""
    if not 16 <= min(width, height) <= max(width, height) <= 16384 or not 1 <= len(regions) <= 256:
        raise ValueError("Measured local geometry exceeds its bounded canvas")
    counts = np.zeros(height, np.int64)
    for index, (x, y, w, h) in enumerate(regions):
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
            raise ValueError("Measured local tile exceeds its original canvas")
        if any(
            min(x + w, px + pw) > max(x, px) and min(y + h, py + ph) > max(y, py)
            for px, py, pw, ph in regions[:index]
        ):
            raise ValueError("Original measured local tiles overlap")
        left, right = max(x, 8), min(x + w, width - 8)
        top, bottom = max(y, 8), min(y + h, height - 8)
        if right > left and bottom > top:
            counts[top:bottom] += right - left
    total = int(counts.sum())
    if not total:
        raise ValueError("Measured local support has no interior occupied area")
    split = int(np.searchsorted(np.cumsum(counts), (total + 1) // 2))
    owners = tuple(int(y + h / 2 >= split) for x, y, w, h in regions)
    if set(owners) != {0, 1}:
        raise ValueError("Measured local support requires two predeclared regions")
    return split, owners


def _owned_point(point: Coordinates, regions: tuple[Region, ...], owners: tuple[int, ...]) -> int:
    x, y = int(float(point[0])), int(float(point[1]))
    found = tuple(
        owner
        for tile, owner in zip(regions, owners, strict=True)
        if tile[0] <= x < tile[0] + tile[2] and tile[1] <= y < tile[1] + tile[3]
    )
    if len(found) != 1:
        raise ValueError("Current point lacks unambiguous measured tile ownership")
    return found[0]


def _local_material(
    selection: SpatialFeatureSelection,
) -> tuple[tuple[int, int, Coordinates, tuple[str, int]], ...]:
    return tuple(
        chain.from_iterable(
            tuple((cohort, index, point, origins[index]) for index, point in enumerate(group))
            for cohort, (group, origins) in enumerate(
                zip(selection.cohorts, selection.origins, strict=True)
            )
        )
    )


def select_current_local_features(
    image: Image,
    regions: tuple[Region, ...],
    *,
    tile_owners: tuple[int, ...],
    seed_split_y: int,
    continuing: tuple[Coordinates, Coordinates] | None = None,
) -> LocalFeatureSelection:
    """Keep the existing bounded selector; group genuine points by seed ownership.

    No fit result selects points or shrinks a failed denominator. The unchanged
    per-cohort cap is applied before any future correspondence is measured.
    """
    if len(regions) != len(tile_owners) or set(tile_owners) != {0, 1}:
        raise ValueError("Current local regions require their original tile owners")
    if not 0 <= seed_split_y < image.shape[0]:
        raise ValueError("Original measured area split exceeds its canvas")
    if any(min(w, h) < 0 for x, y, w, h in regions):
        raise ValueError("Current local tile geometry is invalid")
    material = tuple(tile for tile in regions if min(tile[2], tile[3]) > 0)
    if not material:
        raise ValueError("Current local interior is unavailable")
    selection = select_current_stable_features(image, material, continuing=continuing)
    groups: tuple[list[tuple[float, float]], list[tuple[float, float]]] = ([], [])
    origins: tuple[list[tuple[str, int]], list[tuple[str, int]]] = ([], [])
    source_indices: tuple[list[tuple[int, int]], list[tuple[int, int]]] = ([], [])
    excluded: list[tuple[int, int]] = []
    for cohort, index, point, origin in _local_material(selection):
        owner = _owned_point(point, regions, tile_owners)
        if origin[0] not in {"fresh", f"continued-{owner}"}:
            raise ValueError("Current continued feature changed its measured model owner")
        if len(groups[owner]) >= MAX_TEXTURE_FEATURES // 2:
            excluded.append((cohort, index))
            continue
        groups[owner].append((float(point[0]), float(point[1])))
        origins[owner].append(origin)
        source_indices[owner].append((cohort, index))
    return LocalFeatureSelection(
        (
            np.asarray(groups[0], np.float32).reshape(-1, 2),
            np.asarray(groups[1], np.float32).reshape(-1, 2),
        ),
        (tuple(origins[0]), tuple(origins[1])),
        selection,
        (tuple(source_indices[0]), tuple(source_indices[1])),
        tuple(excluded),
        tile_owners,
        seed_split_y,
    )


@dataclass(frozen=True, slots=True)
class _FeatureCandidate:
    point: Point
    cell: int
    cohort: int | None
    origin: tuple[str, int]


def _region_mask(image: Image, regions: tuple[Region, ...]) -> Image:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected BGR foreground image")
    if not 1 <= len(regions) <= 256:
        raise ValueError("Expected bounded measured interior regions")
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    for x, y, w, h in regions:
        if min(x, y) < 0 or min(w, h) < 1 or x + w > width or y + h > height:
            raise ValueError("Interior region exceeds image")
        mask[y : y + h, x : x + w] = 255
    mask[:8] = mask[-8:] = 0
    mask[:, :8] = mask[:, -8:] = 0
    return mask


def _cell(point: Point, split: tuple[int, int]) -> int:
    return int(point[0] >= split[0]) + 2 * int(point[1] >= split[1])


def _candidates(
    points: Coordinates,
    mask: Image,
    split: tuple[int, int],
    *,
    cohort: int | None,
) -> tuple[_FeatureCandidate, ...]:
    if points.ndim != 2 or points.shape[1] != 2 or len(points) > MAX_TEXTURE_FEATURES:
        raise ValueError("Expected bounded current feature coordinates")
    if not np.isfinite(points).all():
        raise ValueError("Current feature coordinates must be finite")
    height, width = mask.shape
    result: list[_FeatureCandidate] = []
    for index, point in enumerate(points):
        x, y = float(point[0]), float(point[1])
        if 0 <= x < width and 0 <= y < height and mask[int(y), int(x)]:
            location = (x, y)
            result.append(
                _FeatureCandidate(
                    location,
                    _cell(location, split),
                    cohort,
                    ("fresh" if cohort is None else f"continued-{cohort}", index),
                )
            )
    return tuple(result)


def _distributed(candidates: tuple[_FeatureCandidate, ...]) -> tuple[_FeatureCandidate, ...]:
    cells = tuple(_cell_candidates(candidates, cell) for cell in range(4))
    return tuple(point for point in chain.from_iterable(zip_longest(*cells)) if point is not None)


def _cell_candidates(
    candidates: tuple[_FeatureCandidate, ...], cell: int
) -> tuple[_FeatureCandidate, ...]:
    return tuple(point for point in candidates if point.cell == cell)


def _select_candidates(
    candidates: tuple[_FeatureCandidate, ...],
) -> tuple[tuple[Coordinates, Coordinates], tuple[tuple[tuple[str, int], ...], ...]]:
    groups: tuple[list[Point], list[Point]] = ([], [])
    origins: tuple[list[tuple[str, int]], list[tuple[str, int]]] = ([], [])
    selected: list[Point] = []
    per_cell = np.zeros((2, 4), np.int64)
    for candidate in candidates:
        if len(selected) >= MAX_TEXTURE_FEATURES:
            break
        if selected and np.any(
            np.linalg.norm(np.asarray(selected) - candidate.point, axis=1) < FEATURE_SPACING_PIXELS
        ):
            continue
        cohort = candidate.cohort
        if cohort is None:
            cohort = min(range(2), key=lambda i: (per_cell[i, candidate.cell], len(groups[i]), i))
        if len(groups[cohort]) >= MAX_TEXTURE_FEATURES // 2:
            continue
        groups[cohort].append(candidate.point)
        origins[cohort].append(candidate.origin)
        per_cell[cohort, candidate.cell] += 1
        selected.append(candidate.point)
    return (
        (
            np.asarray(groups[0], np.float32).reshape(-1, 2),
            np.asarray(groups[1], np.float32).reshape(-1, 2),
        ),
        (tuple(origins[0]), tuple(origins[1])),
    )


def select_current_stable_features(
    image: Image,
    regions: tuple[Region, ...],
    *,
    continuing: tuple[Coordinates, Coordinates] | None = None,
) -> SpatialFeatureSelection:
    """Distribute genuine current support before the next correspondence is read.

    Continuing endpoints must come from a previously accepted model. They are
    merely next-frame candidates: every point is remeasured with unchanged
    texture/model guards. Empty cells are never filled with invented support.
    """
    mask = _region_mask(image, regions)
    occupied = mask != 0
    total = int(np.count_nonzero(occupied))
    split = (
        int(np.searchsorted(np.cumsum(occupied.sum(axis=0)), (total + 1) // 2)),
        int(np.searchsorted(np.cumsum(occupied.sum(axis=1)), (total + 1) // 2)),
    )
    x, y = split
    area = (
        int(np.count_nonzero(occupied[:y, :x])),
        int(np.count_nonzero(occupied[:y, x:])),
        int(np.count_nonzero(occupied[y:, :x])),
        int(np.count_nonzero(occupied[y:, x:])),
    )
    fresh = _candidates(foreground_features(image, regions), mask, split, cohort=None)
    old = (
        ()
        if continuing is None
        else tuple(
            chain.from_iterable(
                _candidates(group, mask, split, cohort=index)
                for index, group in enumerate(continuing)
            )
        )
    )
    candidates = (*_distributed(old), *_distributed(fresh))
    cohorts, origins = _select_candidates(candidates)
    selected_origins = frozenset(chain.from_iterable(origins))
    continued = tuple(sum(point.cell == cell for point in old) for cell in range(4))
    new = tuple(sum(point.cell == cell for point in fresh) for cell in range(4))
    old_selected = tuple(
        sum(point.cell == cell and point.origin in selected_origins for point in old)
        for cell in range(4)
    )
    fresh_selected = tuple(
        sum(point.cell == cell and point.origin in selected_origins for point in fresh)
        for cell in range(4)
    )
    return SpatialFeatureSelection(
        cohorts,
        area,
        continued,
        new,
        old_selected,
        fresh_selected,
        (origins[0], origins[1]),
        split,
        0 if continuing is None else sum(len(group) for group in continuing) - len(old),
    )


@dataclass(frozen=True, slots=True)
class TextureFit:
    """Actual correspondence fit, including rejected exact-model evidence."""

    source: Coordinates
    metrics: ContourTrackMetrics
    matrix: Matrix | None
    inliers: tuple[bool, ...]
    residuals: tuple[float | None, ...]
    source_span: tuple[float, float]
    inlier_span: tuple[float, float]
    reason: str | None


@dataclass(frozen=True, slots=True)
class TextureMotion:
    contour: Coordinates | None
    features: Coordinates
    inliers: tuple[bool, ...]
    residuals: tuple[float | None, ...]
    metrics: ContourTrackMetrics
    reason: str | None
    fit: TextureFit | None = None


def foreground_features(
    image: Image, regions: tuple[tuple[int, int, int, int], ...]
) -> Coordinates:
    """Select texture only within native-measured fully occupied foreground tiles."""
    mask = _region_mask(image, regions)
    found = cv2.goodFeaturesToTrack(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        maxCorners=256,
        qualityLevel=0.01,
        minDistance=7,
        mask=mask,
        blockSize=7,
    )
    return (
        np.empty((0, 2), np.float32)
        if found is None
        else np.asarray(found.reshape(-1, 2), np.float32)
    )


def fit_texture_model(features: Coordinates, metrics: ContourTrackMetrics) -> TextureFit:
    """Expose one original 1.5-pixel RANSAC fit without changing its guards."""
    if features.ndim != 2 or features.shape[1] != 2 or not 12 <= len(features) <= 256:
        raise ValueError("Expected bounded texture correspondence features")
    if not np.isfinite(features).all() or metrics.points.shape != features.shape:
        raise ValueError("Texture correspondence geometry differs")
    supported = metrics.supported
    empty = tuple(False for _ in features)
    errors: tuple[float | None, ...] = tuple(None for _ in features)
    total_span = np.ptp(features, axis=0)
    source_span = float(total_span[0]), float(total_span[1])
    if int(np.count_nonzero(supported)) < 12:
        return TextureFit(
            features, metrics, None, empty, errors, source_span, (0, 0), "texture_support_lost"
        )
    matrix, fit = cv2.estimateAffinePartial2D(
        features[supported],
        metrics.points[supported],
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or fit is None or not np.isfinite(matrix).all():
        return TextureFit(
            features, metrics, None, empty, errors, source_span, (0, 0), "motion_model_missing"
        )
    bound: Matrix = (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
    )
    predicted = features @ matrix[:, :2].T + matrix[:, 2]
    residual = np.linalg.norm(predicted - metrics.points, axis=1)
    inliers = supported & (residual <= 1.5)
    count = int(np.count_nonzero(inliers))
    reason: str | None = None
    # Model agreement is measured among independently valid correspondences.
    # Missing points still constrain spatial coverage below; they are never inliers.
    if count < 12 or count / int(np.count_nonzero(supported)) < 0.75:
        reason = "motion_support_insufficient"
    else:
        span = np.ptp(features[inliers], axis=0)
        if np.any(total_span < 8) or np.any(span < total_span * 0.5):
            reason = "motion_support_clustered"
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
    if not 0.9 <= scale <= 1.1 or rotation > 15:
        reason = "motion_deformation_exceeded"
    inlier_span = np.ptp(features[inliers], axis=0) if count else np.zeros(2)
    return TextureFit(
        features,
        metrics,
        bound,
        tuple(bool(value) for value in inliers),
        tuple(float(value) if np.isfinite(value) else None for value in residual),
        source_span,
        (float(inlier_span[0]), float(inlier_span[1])),
        reason,
    )


def derive_contour_motion(
    source: Image, donor: Image, features: Coordinates, contour: Coordinates
) -> TextureMotion:
    """Keep the original exact contour contract; expose the same underlying fit."""
    if (
        len(features) < 12
        or contour.ndim != 2
        or contour.shape[1] != 2
        or not np.isfinite(contour).all()
    ):
        raise ValueError("Insufficient texture or invalid measured contour")
    fitted = fit_texture_model(features, measure_contour_tracks(source, donor, features))
    metrics = fitted.metrics
    if fitted.matrix is None:
        return TextureMotion(
            None, metrics.points, fitted.inliers, fitted.residuals, metrics, fitted.reason, fitted
        )
    matrix = np.asarray(fitted.matrix, np.float64)
    reason = fitted.reason
    moved = np.asarray(contour @ matrix[:, :2].T + matrix[:, 2], np.float32)
    if np.any(moved < 0) or np.any(moved >= (source.shape[1], source.shape[0])):
        reason = "derived_contour_clipped"
    return TextureMotion(
        moved if reason is None else None,
        metrics.points,
        fitted.inliers,
        fitted.residuals,
        metrics,
        reason,
        fitted,
    )
