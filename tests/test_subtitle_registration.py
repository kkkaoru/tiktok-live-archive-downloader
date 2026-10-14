"""Known transforms and negative controls for original-only pixel registration."""

from dataclasses import replace
from fractions import Fraction

import cv2
import numpy as np
import pytest

from replay.subtitle_quality_observe import OriginalContext, register


@pytest.fixture
def context() -> OriginalContext:
    noise = np.random.default_rng(19).integers(20, 220, (239, 629, 3), dtype=np.uint8)
    image = np.asarray(cv2.GaussianBlur(noise, (5, 5), 0), dtype=np.uint8)
    mask = np.zeros((239, 629), np.uint8)
    cv2.putText(mask, "SYNTHETIC TEXT", (28, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    cv2.putText(mask, "second row", (210, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    excluded = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    image[excluded] = 255
    return OriginalContext(image, excluded, Fraction(0))


@pytest.mark.parametrize("shift", [(0.0, 0.0), (2.0, -3.0), (0.45, 0.45), (-0.45, -0.45)])
def test_translation_pixel_convention(context: OriginalContext, shift: tuple[float, float]) -> None:
    matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32)
    donor = OriginalContext(
        np.asarray(cv2.warpAffine(context.image, matrix, (629, 239)), dtype=np.uint8),
        cv2.warpAffine(context.excluded.astype(np.float32), matrix, (629, 239)) > 0,
        Fraction(1),
    )
    result = register(context, donor)
    reverse = register(donor, context)
    assert result.inlier_fraction >= 0.9
    assert result.displacement == pytest.approx(shift, abs=0.08)
    assert reverse.displacement == pytest.approx(tuple(-value for value in shift), abs=0.08)
    assert np.any(result.context_geometry)
    expected = np.array([314 + shift[0], 190 + shift[1]])
    np.testing.assert_allclose(result.donor_xy[190, 314], expected, atol=0.08)
    # Bilinear footprints touching even one excluded donor pixel stay unknown.
    contamination = cv2.remap(
        donor.excluded.astype(np.float32),
        result.donor_xy[:, :, 0],
        result.donor_xy[:, :, 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1,
    )
    assert not np.any(result.known & (contamination > 0))


@pytest.mark.parametrize("fault", ["rotation", "deformation", "independent-motion"])
def test_nontranslation_is_not_accepted(context: OriginalContext, fault: str) -> None:
    yy, xx = np.indices(context.excluded.shape, dtype=np.float32)
    if fault == "rotation":
        matrix = cv2.getRotationMatrix2D((314, 119), 2, 1)
        moved = cv2.warpAffine(context.image, matrix, (629, 239))
    else:
        displacement = 3 * np.sin(xx / 40) if fault == "deformation" else np.where(xx < 314, 3, -3)
        moved = cv2.remap(
            context.image,
            np.asarray(xx, dtype=np.float32),
            np.asarray(yy + displacement, np.float32),
            cv2.INTER_LINEAR,
        )
    result = register(context, replace(context, image=np.asarray(moved, np.uint8), pts=Fraction(1)))
    assert result.displacement is None
    assert not np.any(result.context_geometry)


def test_illumination_is_not_corrected_to_a_pass(context: OriginalContext) -> None:
    brighter = np.minimum(context.image.astype(np.int16) + 24, 255).astype(np.uint8)
    result = register(context, replace(context, image=brighter, pts=Fraction(1)))
    assert not np.any(result.known & ~context.excluded)


def test_occlusion_outside_caption_is_unknown(context: OriginalContext) -> None:
    changed = context.image.copy()
    changed[180:195, 300:325] = 0
    result = register(context, replace(context, image=changed, pts=Fraction(1)))
    assert not np.any(result.known[180:195, 300:325])
    assert not np.any(result.context_geometry[180:195, 300:325])


def test_excluded_caption_does_not_set_motion(context: OriginalContext) -> None:
    changed = context.image.copy()
    changed[context.excluded] = 0
    result = register(context, replace(context, image=changed, pts=Fraction(1)))
    assert result.displacement == (0, 0)
    assert result.inlier_fraction == 1
    assert not np.any(result.known & context.excluded)
