"""Scale changes and layout limits do not become face observations."""

from fractions import Fraction

import pytest

from replay.core import ReplayError
from replay.head_near_placement import (
    FaceSizeObservation,
    FaceVerticalExtent,
    HeadNearPlacement,
    caption_only_bottom,
    current_face_heights,
    place_near_head,
)


def test_unknown_geometry_screen_fit_preserves_full_text_without_claiming_face() -> None:
    assert caption_only_bottom(caption_height=600, canvas_height=1920) == 624
    assert caption_only_bottom(caption_height=100, canvas_height=1920) == 124


@pytest.mark.parametrize("height", [0, 2000, float("nan"), True])
def test_unknown_geometry_cannot_hide_invalid_text_size(height: float) -> None:
    with pytest.raises(ReplayError, match="screen-safe"):
        caption_only_bottom(caption_height=height, canvas_height=1920)


def test_single_axis_noise_does_not_shrink_head() -> None:
    reference = FaceSizeObservation(Fraction(0), 100, 200)
    result = current_face_heights(
        (
            reference,
            FaceSizeObservation(Fraction(1, 10), 100, 100),
            FaceSizeObservation(Fraction(2, 10), 50, 200),
            FaceSizeObservation(Fraction(3, 10), 200, 200),
            FaceSizeObservation(Fraction(4, 10), 100, 400),
        ),
        reference=reference,
    )
    assert result == (200, 200, 200, 400, 400)


def test_reset_estimate_uses_both_measured_axes_not_fixed_reference_height() -> None:
    result = current_face_heights(
        (FaceSizeObservation(Fraction(8), 375, 280),),
        reference=FaceSizeObservation(Fraction(0), 400, 320),
    )
    assert result == (300,)


def test_joint_short_contraction_follows_current_geometry() -> None:
    reference = FaceSizeObservation(Fraction(0), 100, 200)
    result = current_face_heights(
        (
            reference,
            FaceSizeObservation(Fraction(1, 10), 50, 100),
            FaceSizeObservation(Fraction(3, 10), 50, 100),
            FaceSizeObservation(Fraction(4, 10), 100, 200),
        ),
        reference=reference,
    )
    assert result == (200, 100, 100, 200)


def test_joint_shrink_never_retains_previous_large_scale() -> None:
    reference = FaceSizeObservation(Fraction(0), 100, 200)
    result = current_face_heights(
        (
            reference,
            FaceSizeObservation(Fraction(1, 10), 50, 80),
            FaceSizeObservation(Fraction(3, 10), 50, 80),
            FaceSizeObservation(Fraction(5, 10), 50, 80),
            FaceSizeObservation(Fraction(6, 10), 50, 80),
            FaceSizeObservation(Fraction(7, 10), 150, 300),
        ),
        reference=reference,
    )
    assert result == (200, 100, 100, 100, 100, 300)


def test_returning_scale_does_not_depend_on_prior_history() -> None:
    reference = FaceSizeObservation(Fraction(0), 100, 200)
    result = current_face_heights(
        (
            reference,
            FaceSizeObservation(Fraction(1, 10), 50, 100),
            FaceSizeObservation(Fraction(3, 10), 100, 200),
            FaceSizeObservation(Fraction(5, 10), 50, 100),
            FaceSizeObservation(Fraction(7, 10), 50, 100),
        ),
        reference=reference,
    )
    assert result == (200, 100, 200, 100, 100)


def test_cut_gap_missing_reset_without_inventing_observation() -> None:
    reference = FaceSizeObservation(Fraction(0), 100, 200)
    result = current_face_heights(
        (
            reference,
            FaceSizeObservation(Fraction(1, 10), 50, 80),
            FaceSizeObservation(Fraction(2, 10), 25, 40, False),
            FaceSizeObservation(Fraction(3, 10), 40, 60),
            FaceSizeObservation(Fraction(1), 80, 120),
        ),
        reference=reference,
    )
    assert result == (200, 100, 50, 80, 160)


def test_derived_scale_changes_extension_only_and_translation_stays_independent() -> None:
    result = place_near_head(
        face=FaceVerticalExtent(600, 700),
        reference_face=FaceVerticalExtent(900, 1300),
        reference_head_top=500,
        caption_height=100,
        canvas_height=1920,
        estimated_face_height=200,
    )
    assert result == HeadNearPlacement(400, 376, 1, "scale-derived")


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), 2000])
def test_bad_derived_height_rejected(value: float) -> None:
    with pytest.raises(ReplayError, match="derived face height"):
        place_near_head(
            face=FaceVerticalExtent(600, 700),
            reference_face=FaceVerticalExtent(900, 1300),
            reference_head_top=500,
            caption_height=100,
            canvas_height=1920,
            estimated_face_height=value,
        )


@pytest.mark.parametrize(
    "observations",
    [
        (),
        (FaceSizeObservation(Fraction(0), 0, 2),),
        (FaceSizeObservation(Fraction(0), 1, float("nan")),),
        (FaceSizeObservation(Fraction(0), 1, 2), FaceSizeObservation(Fraction(0), 1, 2)),
    ],
)
def test_invalid_scale_support_rejected(
    observations: tuple[FaceSizeObservation, ...],
) -> None:
    with pytest.raises(ReplayError):
        current_face_heights(observations, reference=FaceSizeObservation(Fraction(0), 1, 2))


def test_smaller_face_scales_offset_instead_of_leaving_canvas() -> None:
    result = place_near_head(
        face=FaceVerticalExtent(400, 500),
        reference_face=FaceVerticalExtent(900, 1300),
        reference_head_top=500,
        caption_height=200,
        canvas_height=1920,
    )
    assert result == HeadNearPlacement(300, 276, 1, "scale-derived")


def test_larger_face_and_translation_follow_measured_scale() -> None:
    result = place_near_head(
        face=FaceVerticalExtent(1100, 1700),
        reference_face=FaceVerticalExtent(900, 1300),
        reference_head_top=700,
        caption_height=300,
        canvas_height=1920,
    )
    assert result == HeadNearPlacement(800, 776, 0.5, "scale-derived")


def test_canvas_fit_preserves_entire_caption_and_face_clearance() -> None:
    result = place_near_head(
        face=FaceVerticalExtent(400, 800),
        reference_face=FaceVerticalExtent(900, 1300),
        reference_head_top=400,
        caption_height=300,
        canvas_height=1920,
    )
    assert result == HeadNearPlacement(-100, 300, 1.25, "canvas-fitted")


def test_no_room_is_explicit_and_can_be_resolved_by_shorter_reflow() -> None:
    blocked = place_near_head(
        face=FaceVerticalExtent(250, 450),
        reference_face=FaceVerticalExtent(900, 1300),
        reference_head_top=700,
        caption_height=300,
        canvas_height=1920,
    )
    shorter = place_near_head(
        face=FaceVerticalExtent(250, 450),
        reference_face=FaceVerticalExtent(900, 1300),
        reference_head_top=700,
        caption_height=100,
        canvas_height=1920,
    )
    assert blocked == HeadNearPlacement(150, None, 0.5, "insufficient-space-above-face")
    assert shorter == HeadNearPlacement(150, 126, 0.5, "scale-derived")


def test_exact_fit_and_zero_extension() -> None:
    result = place_near_head(
        face=FaceVerticalExtent(324, 500),
        reference_face=FaceVerticalExtent(900, 1300),
        reference_head_top=900,
        caption_height=300,
        canvas_height=1920,
    )
    assert result == HeadNearPlacement(324, 300, 0, "scale-derived")


@pytest.mark.parametrize(
    ("face", "reference", "head", "height", "canvas", "gap"),
    [
        (FaceVerticalExtent(0, 0), FaceVerticalExtent(400, 600), 200, 100, 1920, 24),
        (FaceVerticalExtent(-1, 200), FaceVerticalExtent(400, 600), 200, 100, 1920, 24),
        (FaceVerticalExtent(400, 2000), FaceVerticalExtent(400, 600), 200, 100, 1920, 24),
        (FaceVerticalExtent(400, 600), FaceVerticalExtent(400, 600), 500, 100, 1920, 24),
        (FaceVerticalExtent(400, 600), FaceVerticalExtent(400, 400), 200, 100, 1920, 24),
        (FaceVerticalExtent(400, 600), FaceVerticalExtent(400, 600), -1, 100, 1920, 24),
        (FaceVerticalExtent(400, 600), FaceVerticalExtent(400, 600), 200, 0, 1920, 24),
        (FaceVerticalExtent(400, 600), FaceVerticalExtent(400, 600), 200, 2000, 1920, 24),
        (FaceVerticalExtent(400, 600), FaceVerticalExtent(400, 600), 200, 100, 1920, -1),
        (FaceVerticalExtent(float("nan"), 600), FaceVerticalExtent(400, 600), 200, 100, 1920, 24),
        (FaceVerticalExtent(400, 600), FaceVerticalExtent(400, 600), True, 100, 1920, 24),
    ],
)
def test_invalid_geometry_rejected(
    face: FaceVerticalExtent,
    reference: FaceVerticalExtent,
    head: float,
    height: float,
    canvas: float,
    gap: float,
) -> None:
    with pytest.raises(ReplayError, match="finite face"):
        place_near_head(
            face=face,
            reference_face=reference,
            reference_head_top=head,
            caption_height=height,
            canvas_height=canvas,
            gap=gap,
        )
