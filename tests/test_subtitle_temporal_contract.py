"""Separate apparent temporal change from restored background and artifact cause."""

from dataclasses import replace
from fractions import Fraction

import cv2
import numpy as np
import pytest

from replay import subtitle_quality_observe as observe
from replay.delivery_mask_quality import LosslessFrame, measure_batch, summarize_quality
from replay.subtitle_clock import VideoFrameClock

CLOCK = VideoFrameClock((0, 1, 2), Fraction(1, 30), 1)


@pytest.fixture
def pair() -> tuple[observe.ObservationFrame, observe.ObservationFrame]:
    background = np.random.default_rng(42).integers(20, 180, (192, 192, 3), dtype=np.uint8)
    repair = np.zeros((192, 192), np.bool_)
    repair[90:100, 90:100] = True
    source = background.copy()
    source[repair] = 255
    first = observe.ObservationFrame(
        observe.OriginalContext(source, repair, Fraction(0)),
        background,
        repair,
        repair,
        None,
        None,
    )
    second = replace(first, original=replace(first.original, pts=Fraction(1, 30)))
    return first, second


def test_static_unknown_background_is_not_reconstruction(
    pair: tuple[observe.ObservationFrame, observe.ObservationFrame],
) -> None:
    first, second = pair
    result = observe.candidate_pair_change(first, second, clock=CLOCK)
    assert result.changed_fraction == 0
    assert result.registration is not None
    assert result.registration.displacement == (0, 0)
    assert np.count_nonzero(result.registration.known & first.repair) == 0
    assert np.count_nonzero(result.registration.context_geometry & first.repair) == 100
    # The old boundary gate remains unknown, despite an observed outgoing pair.
    assert not observe.temporal_observation(first).evaluated


def test_source_registered_translation(
    pair: tuple[observe.ObservationFrame, observe.ObservationFrame],
) -> None:
    first, second = pair
    matrix = np.array([[1, 0, 2], [0, 1, 0]], dtype=np.float32)
    second = replace(
        second,
        original=replace(
            second.original,
            image=np.asarray(
                cv2.warpAffine(second.original.image, matrix, (192, 192)), dtype=np.uint8
            ),
            excluded=cv2.warpAffine(second.repair.astype(np.uint8), matrix, (192, 192)) > 0,
        ),
        candidate=np.asarray(cv2.warpAffine(second.candidate, matrix, (192, 192)), dtype=np.uint8),
        glyph=cv2.warpAffine(second.glyph.astype(np.uint8), matrix, (192, 192)) > 0,
        repair=cv2.warpAffine(second.repair.astype(np.uint8), matrix, (192, 192)) > 0,
    )
    result = observe.candidate_pair_change(first, second, clock=CLOCK)
    assert result.changed_fraction == 0
    assert result.registration is not None
    assert result.registration.displacement == pytest.approx((2, 0), abs=0.01)


@pytest.mark.parametrize("codes, expected", [(6, 0.0), (20, None)])
def test_natural_global_light_not_silently_normalized(
    pair: tuple[observe.ObservationFrame, observe.ObservationFrame],
    codes: int,
    expected: float | None,
) -> None:
    first, second = pair
    source = np.minimum(second.original.image.astype(np.int16) + codes, 255).astype(np.uint8)
    candidate = np.minimum(second.candidate.astype(np.int16) + codes, 255).astype(np.uint8)
    result = observe.candidate_pair_change(
        first,
        replace(second, original=replace(second.original, image=source), candidate=candidate),
        clock=CLOCK,
    )
    assert result.changed_fraction == expected


@pytest.mark.parametrize("cause", ["flicker", "hidden-local-light", "hidden-occlusion"])
def test_identical_hidden_causes_are_not_distinguishable(
    pair: tuple[observe.ObservationFrame, observe.ObservationFrame], cause: str
) -> None:
    first, second = pair
    changed = second.candidate.copy()
    changed[second.repair] = np.minimum(changed[second.repair].astype(np.int16) + 40, 255)
    # All three causes have exactly the same observed original (occluded by text).
    # No function of these inputs can truthfully distinguish their causes.
    assert cause in {"flicker", "hidden-local-light", "hidden-occlusion"}
    result = observe.candidate_pair_change(first, replace(second, candidate=changed), clock=CLOCK)
    assert result.changed_fraction == 1
    assert result.repair_pixels == 100
    assert np.count_nonzero(result.unknown) == 0
    assert not hasattr(result, "accepted")
    assert not hasattr(result, "unstable_fraction")


def test_visible_occlusion_remains_unknown(
    pair: tuple[observe.ObservationFrame, observe.ObservationFrame],
) -> None:
    first, second = pair
    source = second.original.image.copy()
    source[88:102, 88:102] = 0
    result = observe.candidate_pair_change(
        first, replace(second, original=replace(second.original, image=source)), clock=CLOCK
    )
    assert result.changed_fraction is None
    # Observed mismatch invalidates the 3px border, not the hidden interior.
    assert np.count_nonzero(result.unknown) == 84


def test_constant_residual_has_zero_variation_not_absence(
    pair: tuple[observe.ObservationFrame, observe.ObservationFrame],
) -> None:
    first, second = pair
    result = observe.candidate_pair_change(
        replace(first, candidate=first.original.image),
        replace(second, candidate=second.original.image),
        clock=CLOCK,
    )
    assert result.changed_fraction == 0
    assert np.count_nonzero(first.glyph) == 100
    assert np.array_equal(first.original.image, second.original.image)
    evidence = measure_batch(
        (
            LosslessFrame(
                first.original.image,
                first.original.image,
                first.glyph,
                first.repair,
                result.changed,
                True,
                Fraction(0),
                Fraction(0),
            ),
        ),
        detect=lambda images: (first.glyph.astype(np.uint8) * 255,),
    )
    assert evidence[0].residual_fraction == 1
    assert summarize_quality(evidence, expected=1)["accepted"] is False


def test_constant_wrong_fill_fails_old_donor_metric(
    pair: tuple[observe.ObservationFrame, observe.ObservationFrame],
) -> None:
    first, second = pair
    wrong = second.candidate.copy()
    wrong[second.repair] = 255
    clean = replace(first.original, image=first.candidate, excluded=np.zeros_like(first.repair))
    old = observe.temporal_observation(
        replace(
            second,
            candidate=wrong,
            previous=clean,
            following=replace(clean, pts=Fraction(2, 30)),
        )
    )
    assert old.evaluated
    assert np.count_nonzero(old.unstable) == 100
    assert (
        observe.candidate_pair_change(
            replace(first, candidate=wrong), replace(second, candidate=wrong), clock=CLOCK
        ).changed_fraction
        == 0
    )


def test_local_model_never_fits_candidate_pixels(
    pair: tuple[observe.ObservationFrame, observe.ObservationFrame],
) -> None:
    first, second = pair
    baseline = observe.candidate_pair_change(
        first, second, clock=CLOCK, registration_model="local-affine"
    )
    altered = second.candidate.copy()
    altered[second.repair] = 0
    measured = observe.candidate_pair_change(
        first, replace(second, candidate=altered), clock=CLOCK, registration_model="local-affine"
    )
    assert baseline.registration is not None
    assert measured.registration is not None
    assert measured.registration.local_models_accepted > 0
    np.testing.assert_array_equal(measured.registration.donor_xy, baseline.registration.donor_xy)
    np.testing.assert_array_equal(measured.unknown, baseline.unknown)
    assert not np.any(measured.registration.known & first.repair)
    # The clean-donor metric cannot use subtitle-covered donors as truth.
    center = replace(
        second, previous=first.original, following=replace(second.original, pts=Fraction(2, 30))
    )
    truth = observe.temporal_observation(center, registration_model="local-affine")
    assert not truth.evaluated
    assert np.count_nonzero(truth.unknown) == 100


def test_missing_registration_empty_and_nonadjacent(
    pair: tuple[observe.ObservationFrame, observe.ObservationFrame],
) -> None:
    first, second = pair
    assert observe.candidate_pair_change(first, None, clock=CLOCK).changed_fraction is None
    empty = replace(first, glyph=np.zeros_like(first.glyph), repair=np.zeros_like(first.repair))
    assert observe.candidate_pair_change(empty, second, clock=CLOCK).changed_fraction is None
    untextured = replace(first.original, image=np.zeros_like(first.candidate))
    result = observe.candidate_pair_change(replace(first, original=untextured), second, clock=CLOCK)
    assert result.changed_fraction is None
    assert np.count_nonzero(result.unknown) == 100
    with pytest.raises(ValueError, match="adjacent"):
        observe.candidate_pair_change(second, first, clock=CLOCK)
    with pytest.raises(ValueError, match="adjacent"):
        observe.candidate_pair_change(
            first,
            replace(second, original=replace(second.original, pts=Fraction(2, 30))),
            clock=CLOCK,
        )
    with pytest.raises(ValueError, match="absent"):
        observe.candidate_pair_change(
            replace(first, original=replace(first.original, pts=Fraction(9))), None, clock=CLOCK
        )
