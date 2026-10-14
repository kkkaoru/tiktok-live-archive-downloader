from fractions import Fraction
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from replay.head_contour_track import ContourSeed, track_texture_contour
from replay.head_texture_track import (
    derive_contour_motion,
    fit_texture_model,
    foreground_features,
    measured_region_owners,
    measured_tile_partition,
    select_current_local_features,
    select_current_stable_features,
)
from replay.recordings import record
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image


@pytest.fixture
def image() -> Image:
    raw = np.random.default_rng(52).integers(0, 255, (160, 160, 3), dtype=np.uint8)
    return np.asarray(cv2.GaussianBlur(raw, (3, 3), 0), np.uint8)


@pytest.mark.parametrize("angle", [0, 2, -3])
def test_texture_translation_rotation(image: Image, angle: int) -> None:
    features = foreground_features(image, ((24, 24, 112, 112),))
    contour = np.array([[40, 20], [80, 16], [120, 20]], np.float32)
    matrix = cv2.getRotationMatrix2D((80, 80), angle, 1)
    matrix[:, 2] += (1, 1)
    donor = np.asarray(cv2.warpAffine(image, matrix, (160, 160)), np.uint8)
    motion = derive_contour_motion(image, donor, features, contour)
    assert motion.reason is None
    assert motion.contour is not None
    np.testing.assert_allclose(motion.contour, contour @ matrix[:, :2].T + matrix[:, 2], atol=0.5)


def test_blank_and_occlusion_are_unknown(image: Image) -> None:
    features = foreground_features(image, ((24, 24, 112, 112),))
    contour = np.array([[40, 20], [80, 16], [120, 20]], np.float32)
    assert len(foreground_features(np.zeros_like(image), ((24, 24, 112, 112),))) == 0
    for donor in (np.zeros_like(image), image.copy()):
        donor[:, :100] = 0
        result = derive_contour_motion(image, donor, features, contour)
        assert result.contour is None
        assert result.reason is not None


@pytest.mark.parametrize("regions", [(), ((-1, 0, 10, 10),), ((0, 0, 200, 20),)])
def test_invalid_regions(image: Image, regions: tuple[tuple[int, int, int, int], ...]) -> None:
    with pytest.raises(ValueError):
        foreground_features(image, regions)


def test_invalid_image_and_features(image: Image) -> None:
    with pytest.raises(ValueError):
        foreground_features(image[:, :, :2], ((24, 24, 112, 112),))
    with pytest.raises(ValueError):
        derive_contour_motion(
            image, image, np.zeros((3, 2), np.float32), np.zeros((3, 2), np.float32)
        )


def test_missing_model(image: Image, monkeypatch: pytest.MonkeyPatch) -> None:
    features = foreground_features(image, ((24, 24, 112, 112),))
    monkeypatch.setattr(cv2, "estimateAffinePartial2D", Mock(return_value=(None, None)))
    result = derive_contour_motion(image, image, features, features[:3])
    assert result.reason == "motion_model_missing"
    assert result.fit is not None
    assert result.fit.matrix is None


def test_actual_fit_exposes_rejected_geometry_without_promoting_contour(image: Image) -> None:
    features = foreground_features(image, ((24, 24, 112, 112),))
    contour = np.array([[40, 20], [80, 16], [120, 20]], np.float32)
    motion = derive_contour_motion(image, image, features, contour)
    assert motion.fit is not None
    assert motion.fit.matrix is not None
    assert motion.fit.reason == motion.reason
    assert motion.fit.inliers == motion.inliers
    assert motion.fit.residuals == motion.residuals
    np.testing.assert_array_equal(motion.fit.metrics.points, motion.features)
    assert min(motion.fit.source_span) >= 8


@pytest.mark.parametrize("fault", ["shape", "count", "finite", "metrics"])
def test_typed_fit_rejects_unbound_correspondences(image: Image, fault: str) -> None:
    original = foreground_features(image, ((24, 24, 112, 112),))
    motion = derive_contour_motion(image, image, original, original[:3])
    invalid = original.copy()
    if fault == "shape":
        invalid = original[:, :1]
    elif fault == "count":
        invalid = original[:3]
    elif fault == "finite":
        invalid[0, 0] = np.nan
    else:
        invalid = original[:-1]
    with pytest.raises(ValueError):
        fit_texture_model(invalid, motion.metrics)


def test_texture_clock_and_derived_provenance(image: Image) -> None:
    points = np.array([[40, 20], [80, 16], [120, 20]], np.float32)
    seed = ContourSeed(Fraction(1), 160, 160, points, ((24, 24, 112, 112),))
    clock = VideoFrameClock((30, 31), Fraction(1, 30), 1)
    result = track_texture_contour(seed, clock, ((30, image), (31, image)))
    assert result[1].provenance == "texture-derived-foreground-contour"
    assert result[1].candidate_top == pytest.approx(16)
    assert result[1].reason is None
    with pytest.raises(ValueError, match="incomplete"):
        track_texture_contour(seed, clock, ((30, image),))
    with pytest.raises(ValueError, match="clock"):
        track_texture_contour(seed, clock, ((31, image),))
    with pytest.raises(ValueError, match="contract"):
        track_texture_contour(seed, VideoFrameClock((0, 1), Fraction(1), 1), ())
    blank = track_texture_contour(seed, clock, ((30, np.zeros_like(image)),))
    assert blank[0].reason == "insufficient_foreground_texture"
    lost = track_texture_contour(seed, clock, ((30, image), (31, np.zeros_like(image))))
    assert lost[1].candidate_top is None


def test_seed_anchor_does_not_accumulate_transforms(image: Image) -> None:
    points = np.array([[40, 20], [80, 16], [120, 20]], np.float32)
    seed = ContourSeed(Fraction(1), 160, 160, points, ((24, 24, 112, 112),))
    clock = VideoFrameClock((30, 31, 32), Fraction(1, 30), 1)
    frames = tuple(
        (
            30 + i,
            np.asarray(
                cv2.warpAffine(image, np.array([[1, 0, i * 2], [0, 1, 0]], np.float32), (160, 160)),
                np.uint8,
            ),
        )
        for i in range(3)
    )
    result = track_texture_contour(seed, clock, frames)
    assert result[-1].reason is None
    np.testing.assert_allclose(result[-1].points, points + (4, 0), atol=0.5)


@pytest.mark.parametrize(
    "matrix,reason",
    [
        (np.array([[1.3, 0, 0], [0, 1.3, 0]]), "motion_deformation_exceeded"),
        (np.array([[1, 0, -200], [0, 1, 0]]), "derived_contour_clipped"),
    ],
)
def test_implausible_transform_is_unknown(
    image: Image,
    monkeypatch: pytest.MonkeyPatch,
    matrix: NDArray[np.float64],
    reason: str,
) -> None:
    features = foreground_features(image, ((24, 24, 112, 112),))
    monkeypatch.setattr(
        cv2, "estimateAffinePartial2D", Mock(return_value=(matrix, np.ones(len(features))))
    )
    result = derive_contour_motion(image, image, features, features[:3])
    assert result.reason == reason
    assert result.contour is None


def test_spatial_selection_keeps_current_supported_endpoints_and_excludes_ui(image: Image) -> None:
    regions = ((24, 48, 112, 80),)
    points = foreground_features(image, regions)
    contour = np.array([[40, 20], [80, 16], [120, 20]], np.float32)
    current = np.asarray(
        cv2.warpAffine(image, np.array([[1, 0, 1], [0, 1, 1]], np.float32), (160, 160)), np.uint8
    )
    first = derive_contour_motion(image, current, points[::2].copy(), contour)
    second = derive_contour_motion(image, current, points[1::2].copy(), contour)
    assert first.reason is None
    assert second.reason is None
    continuing = (
        first.features[np.asarray(first.inliers)],
        second.features[np.asarray(second.inliers)],
    )
    current[:32] = 255
    current[144:] = 255
    selection = select_current_stable_features(current, ((25, 49, 112, 80),), continuing=continuing)
    assert sum(selection.selected_continued) >= 24
    assert min(len(selection.cohorts[0]), len(selection.cohorts[1])) >= 12
    assert not set(map(tuple, selection.cohorts[0])).intersection(map(tuple, selection.cohorts[1]))
    assert np.all(selection.cohorts[0][:, 1] >= 49)
    assert np.all(selection.cohorts[1][:, 1] < 129)
    assert min(selection.selected_continued) > 0
    assert selection.audit()["selectionBeforeNextFrame"] is True
    assert selection.audit()["newFaceObservation"] is False


def test_spatial_selection_preserves_stable_upper_points_over_fresh_lower_contrast(
    image: Image,
) -> None:
    regions = ((24, 24, 112, 112),)
    original = select_current_stable_features(image, regions)
    current = image.copy()
    current[88:136, 24:136] = np.random.default_rng(31).integers(
        0, 255, (48, 112, 3), dtype=np.uint8
    )
    stable = (
        original.cohorts[0][original.cohorts[0][:, 1] < 80],
        original.cohorts[1][original.cohorts[1][:, 1] < 80],
    )
    refreshed = select_current_stable_features(current, regions, continuing=stable)
    assert sum(refreshed.selected_continued) == len(stable[0]) + len(stable[1])
    assert sum(refreshed.selected_fresh) > 0
    assert set(map(tuple, stable[0])).issubset(map(tuple, refreshed.cohorts[0]))
    assert set(map(tuple, stable[1])).issubset(map(tuple, refreshed.cohorts[1]))
    assert np.ptp(refreshed.cohorts[0][:, 1]) > 80
    assert np.ptp(refreshed.cohorts[1][:, 1]) > 80


def test_spatial_selection_cannot_fill_blank_support_or_tile_gaps(image: Image) -> None:
    blank = select_current_stable_features(np.zeros_like(image), ((24, 24, 112, 112),))
    assert blank.audit()["selectedFeatureCounts"] == [0, 0]
    gapped = select_current_stable_features(image, ((24, 24, 24, 112), (112, 24, 24, 112)))
    assert np.all((gapped.cohorts[0][:, 0] < 48) | (gapped.cohorts[0][:, 0] >= 112))
    assert np.all((gapped.cohorts[1][:, 0] < 48) | (gapped.cohorts[1][:, 0] >= 112))
    tiny = select_current_stable_features(image, ((0, 0, 4, 4),))
    assert tiny.occupied_pixels == (0, 0, 0, 0)
    assert tiny.audit()["selectedFeatureCounts"] == [0, 0]


def test_spatial_selection_rejects_old_points_outside_current_support(image: Image) -> None:
    old = (
        np.asarray(((2, 2), (90, 90), (100, 100)), np.float32),
        np.asarray(((2, 2), (90, 90), (100, 100)), np.float32),
    )
    result = select_current_stable_features(image, ((24, 24, 24, 24),), continuing=old)
    assert result.rejected_continued == 6
    assert result.selected_continued == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "points",
    [np.zeros((3, 3), np.float32), np.zeros((257, 2), np.float32), np.full((3, 2), np.nan)],
)
def test_spatial_selection_rejects_unbounded_or_nonfinite_history(
    image: Image, points: NDArray[np.float32]
) -> None:
    with pytest.raises(ValueError, match="[Cc]urrent feature"):
        select_current_stable_features(image, ((24, 24, 112, 112),), continuing=(points, points))


def test_spatial_selection_keeps_feature_budget_and_independent_cohorts() -> None:
    image = np.random.default_rng(8).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    fresh = select_current_stable_features(image, ((16, 16, 368, 368),))
    current = select_current_stable_features(image, ((16, 16, 368, 368),), continuing=fresh.cohorts)
    assert len(current.cohorts[0]) == 128
    assert len(current.cohorts[1]) == 128
    assert not set(map(tuple, current.cohorts[0])).intersection(map(tuple, current.cohorts[1]))
    assert sum(current.selected_fresh) == 0


def test_local_ownership_is_measured_before_fit_and_excludes_static_competitors(
    image: Image,
) -> None:
    regions = ((24, 24, 112, 32), (24, 96, 112, 32))
    split, owners = measured_region_owners(image, regions)
    selected = select_current_local_features(image, regions, tile_owners=owners, seed_split_y=split)
    audit = selected.audit()
    assert split == 55
    assert owners == (0, 1)
    assert len(selected.cohorts[0]) >= 12
    assert len(selected.cohorts[1]) >= 12
    assert np.all((selected.cohorts[0][:, 1] >= 24) & (selected.cohorts[0][:, 1] < 56))
    assert np.all((selected.cohorts[1][:, 1] >= 96) & (selected.cohorts[1][:, 1] < 128))
    local = record(audit["predeclaredLocalSupport"])
    assert local["policy"] == "predeclared-measured-region-spatial-v1"
    assert local["fitDependentSelection"] is False
    assert selected.budget_excluded == ()
    assert audit["newFaceObservation"] is False
    refreshed = select_current_local_features(
        image, regions, tile_owners=owners, seed_split_y=split, continuing=selected.cohorts
    )
    assert sum(refreshed.original_selection.selected_continued) > 0
    assert not set(map(tuple, refreshed.cohorts[0])).intersection(map(tuple, refreshed.cohorts[1]))


@pytest.mark.parametrize(
    ("regions", "width", "height"),
    [
        ((), 160, 160),
        (((24, 24, 16, 16),), 160, 160),
        (((0, 0, 4, 4),), 160, 160),
        (((24, 24, 32, 32), (40, 40, 32, 32)), 160, 160),
        (((150, 24, 32, 32), (24, 96, 32, 32)), 160, 160),
        (((24, 24, 32, 32), (24, 96, 32, 32)), 8, 160),
        (((24, 24, 32, 32), (24, 96, 32, 32)), 160, 20000),
    ],
)
def test_local_partition_rejects_unsupported_or_ambiguous_measured_geometry(
    regions: tuple[tuple[int, int, int, int], ...], width: int, height: int
) -> None:
    with pytest.raises(ValueError):
        measured_tile_partition(regions, width=width, height=height)


@pytest.mark.parametrize("fault", ["owners", "split", "negative", "empty", "changed-owner"])
def test_local_selection_never_invents_owner_support(image: Image, fault: str) -> None:
    regions = ((24, 24, 112, 32), (24, 96, 112, 32))
    owners = (0, 1) if fault != "owners" else (0, 0)
    split = 55 if fault != "split" else 160
    continuing = None
    if fault == "negative":
        regions = ((24, 24, -1, 32), (24, 96, 112, 32))
    elif fault == "empty":
        regions = ((24, 24, 0, 0), (24, 96, 0, 0))
    elif fault == "changed-owner":
        continuing = (np.asarray(((40, 110),), np.float32), np.empty((0, 2), np.float32))
    with pytest.raises(ValueError):
        select_current_local_features(
            image, regions, tile_owners=owners, seed_split_y=split, continuing=continuing
        )


def test_local_owner_cap_is_applied_before_fit_and_exclusions_remain_explicit() -> None:
    raw = np.random.default_rng(25).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    regions = ((16, 16, 368, 352), (16, 376, 16, 8))
    selected = select_current_local_features(raw, regions, tile_owners=(0, 1), seed_split_y=190)
    assert len(selected.cohorts[0]) == 128
    assert len(selected.budget_excluded) > 0
    assert len(selected.cohorts[0]) + len(selected.cohorts[1]) + len(
        selected.budget_excluded
    ) == sum(len(group) for group in selected.original_selection.cohorts)


def test_local_projected_tiles_preserve_clipped_empty_identity_and_reject_overlap(
    image: Image,
) -> None:
    selected = select_current_local_features(
        image,
        ((24, 24, 112, 32), (24, 96, 112, 32), (8, 8, 0, 0)),
        tile_owners=(0, 1, 0),
        seed_split_y=55,
    )
    assert min(len(group) for group in selected.cohorts) >= 12
    with pytest.raises(ValueError, match="unambiguous"):
        select_current_local_features(
            image,
            ((24, 24, 112, 112), (24, 24, 112, 112)),
            tile_owners=(0, 1),
            seed_split_y=80,
        )
