"""Held-out local models cannot turn unsupported coordinates into observations."""

from dataclasses import replace
from fractions import Fraction
from itertools import cycle

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from replay.subtitle_quality_observe import OriginalContext, register
from replay.subtitle_registration import local_affine_geometry


@pytest.fixture
def points() -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    yy, xx = np.mgrid[24:168:12, 24:168:12]
    return np.stack((xx.ravel(), yy.ravel()), axis=-1).astype(np.float32)


def test_affine_held_out_interpolation(
    points: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
) -> None:
    targets = points @ np.array([[1, 0.02], [-0.02, 1]], np.float32) + (2, -1)
    result = local_affine_geometry(
        points,
        targets=targets.astype(np.float32),
        valid=np.ones(len(points), np.bool_),
        shape=(192, 192),
    )
    assert result.accepted > 0
    assert result.diagnostics.valid_tracks == 144
    assert result.diagnostics.invalid_tracks == 0
    assert result.diagnostics.invalid_neighbors == 0
    assert result.diagnostics.residual == 0
    assert result.supported[96, 96]
    assert not result.supported[0, 0]
    np.testing.assert_allclose(result.donor_xy[96, 96], [96.08, 96.92], atol=0.001)


@pytest.mark.parametrize(
    "fault", ["inconsistent", "invalid-tracks", "collinear", "too-few", "too-distant"]
)
def test_negatives(points: np.ndarray[tuple[int, ...], np.dtype[np.float32]], fault: str) -> None:
    valid = np.ones(len(points), np.bool_)
    targets = points.copy()
    shape = (192, 192)
    if fault == "inconsistent":
        targets += np.random.default_rng(7).normal(0, 5, points.shape).astype(np.float32)
    elif fault == "invalid-tracks":
        valid[:] = False
    elif fault == "collinear":
        points[:, 1] = 70
        targets = points.copy()
    elif fault == "too-few":
        points, targets, valid = points[:8], targets[:8], valid[:8]
    else:
        points *= 5
        targets = points.copy()
        shape = (800, 800)
    result = local_affine_geometry(points, targets=targets, valid=valid, shape=shape)
    assert not np.any(result.supported)
    assert result.accepted == 0
    diagnostics = result.diagnostics
    assert result.tested == (
        diagnostics.invalid_neighbors + diagnostics.rank + diagnostics.residual + diagnostics.hull
    )
    assert len(points) == (
        result.tested + diagnostics.insufficient_neighbors + diagnostics.distant_neighbors
    )


def test_invalid_neighbor_veto_is_observable_without_residual_selection() -> None:
    yy, xx = np.mgrid[40:80:10, 40:80:10]
    points = np.stack((xx.ravel(), yy.ravel()), axis=-1).astype(np.float32)
    valid = np.ones(16, np.bool_)
    valid[0] = False
    targets = points.copy()
    targets[0] = np.nan
    result = local_affine_geometry(points, targets=targets, valid=valid, shape=(128, 128))
    assert result.tested == 16
    assert result.accepted == 0
    assert result.diagnostics.valid_tracks == 15
    assert result.diagnostics.invalid_tracks == 1
    assert result.diagnostics.invalid_neighbors == 16
    assert result.diagnostics.residual == 0
    assert result.diagnostics.rank == 0
    assert result.diagnostics.hull == 0
    assert not np.any(result.supported)


def test_nonrigid_residual_rejections_are_counted() -> None:
    yy, xx = np.mgrid[40:80:10, 40:80:10]
    points = np.stack((xx.ravel(), yy.ravel()), axis=-1).astype(np.float32)
    targets = points.copy()
    targets[:, 1] += np.square(points[:, 0] - 55) / 20
    result = local_affine_geometry(
        points, targets=targets, valid=np.ones(16, np.bool_), shape=(128, 128)
    )
    assert result.tested == 16
    assert result.accepted == 0
    assert result.diagnostics.invalid_neighbors == 0
    assert result.diagnostics.residual == 16
    assert result.diagnostics.training_residual == 16
    assert result.diagnostics.held_residual == 16
    assert result.diagnostics.rank == 0
    assert result.diagnostics.hull == 0
    assert not np.any(result.supported)


def test_conflicting_neighborhoods_do_not_average_to_a_pass(
    points: np.ndarray[tuple[int, ...], np.dtype[np.float32]],
) -> None:
    # Smooth non-affine motion: a local linear approximation may fit, but does
    # not authorize extrapolation outside both independently supported hulls.
    targets = points.copy()
    targets[:, 1] += 2 * np.sin(points[:, 0] / 18)
    result = local_affine_geometry(
        points, targets=targets, valid=np.ones(len(points), np.bool_), shape=(192, 192)
    )
    assert not result.supported[180, 96]
    assert result.accepted < result.tested


@pytest.mark.parametrize("fault", ["dtype", "nan", "shape", "valid", "canvas"])
def test_malformed(points: np.ndarray[tuple[int, ...], np.dtype[np.float32]], fault: str) -> None:
    targets = points.copy()
    valid = np.ones(len(points), np.bool_)
    shape = (192, 192)
    if fault == "dtype":
        targets = targets.astype(np.float64)
    elif fault == "nan":
        targets[0, 0] = np.nan
    elif fault == "shape":
        targets = targets[:8]
    elif fault == "valid":
        valid = valid[:8]
    else:
        shape = (2, 2)
    with pytest.raises(ValueError, match="geometry"):
        local_affine_geometry(points, targets=targets, valid=valid, shape=shape)


def test_three_overlapping_models_check_all_disagreements(
    points: NDArray[np.float32], monkeypatch: pytest.MonkeyPatch
) -> None:
    offsets = cycle((0.0, 0.4, -0.4))

    def fit(
        _design: NDArray[np.float64], _targets: NDArray[np.float32], rcond: None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int, NDArray[np.float64]]:
        assert rcond is None
        matrix = np.array([[1, 0], [0, 1], [next(offsets), 0]], np.float64)
        return matrix, np.zeros(0, np.float64), 3, np.ones(3, np.float64)

    monkeypatch.setattr(np.linalg, "lstsq", fit)
    result = local_affine_geometry(
        points, targets=points.copy(), valid=np.ones(len(points), np.bool_), shape=(192, 192)
    )
    assert result.accepted > 0
    # Each model passes 0.5px held-out checks, and both +/-0.4 are close to 0,
    # but their 0.8px mutual disagreement must invalidate overlapping support.
    assert not result.supported[96, 96]
    assert result.diagnostics.agreement_pixels > 0


def test_source_only_local_registration() -> None:
    image = np.random.default_rng(12).integers(20, 220, (256, 256, 3), dtype=np.uint8)
    image = np.asarray(cv2.GaussianBlur(image, (5, 5), 0), dtype=np.uint8)
    excluded = np.zeros((256, 256), np.bool_)
    original = OriginalContext(image, excluded, Fraction(0))
    matrix = cv2.getRotationMatrix2D((128, 128), 1, 1)
    donor = replace(
        original,
        image=np.asarray(cv2.warpAffine(image, matrix, (256, 256)), np.uint8),
        pts=Fraction(1),
    )
    global_result = register(original, donor)
    result = register(original, donor, model="local-affine")
    assert global_result.displacement is None
    assert result.local_models_accepted > 0
    assert result.local_diagnostics is not None
    assert result.local_diagnostics.valid_tracks > 16
    assert np.count_nonzero(result.known) > 1000
    assert not result.context_geometry[0, 0]
    # Bad source appearance still invalidates local geometry after model fitting.
    lit = replace(donor, image=np.minimum(donor.image.astype(np.int16) + 40, 255).astype(np.uint8))
    assert not np.any(register(original, lit, model="local-affine").known)
