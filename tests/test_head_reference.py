from dataclasses import replace
from fractions import Fraction

import pytest
from test_fcp_head_near import paired as paired

from replay.core import ReplayError
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_tracking_caption import MissingFaceRange, TrackingPosition
from replay.head_reference import automatic_reference, resolve_reference


def test_automatic_reference_uses_shared_estimate(paired: FaceScaleSeries) -> None:
    result = resolve_reference({"headReferencePolicy": "automatic-face-relative"}, paired)
    assert result.reference_time == 0
    assert result.source_head_top == 400
    assert result.roi == (0, 900, 1080, 400)


def test_automatic_reference_scales_and_clamps(paired: FaceScaleSeries) -> None:
    smaller = FaceScaleSeries(
        replace(
            paired.tops,
            positions=(
                TrackingPosition(Fraction(0), 0, 450),
                TrackingPosition(Fraction(1), 0, 200),
            ),
        ),
        replace(
            paired.bottoms,
            positions=(
                TrackingPosition(Fraction(0), 0, 650),
                TrackingPosition(Fraction(1), 0, 250),
            ),
        ),
    )
    assert automatic_reference(smaller).source_head_top == 200
    crowded = FaceScaleSeries(
        replace(
            paired.tops,
            positions=(
                TrackingPosition(Fraction(0), 0, 100),
                TrackingPosition(Fraction(1), 0, 120),
            ),
        ),
        replace(
            paired.bottoms,
            positions=(
                TrackingPosition(Fraction(0), 0, 500),
                TrackingPosition(Fraction(1), 0, 520),
            ),
        ),
    )
    assert automatic_reference(crowded).source_head_top == 0


def test_automatic_reference_excludes_holds(paired: FaceScaleSeries) -> None:
    gaps = (
        MissingFaceRange(
            Fraction(0), Fraction(1), 1, Fraction(1), "leading-observation-backfill", "missing"
        ),
    )
    held = FaceScaleSeries(
        replace(paired.tops, derived_ranges=gaps), replace(paired.bottoms, derived_ranges=gaps)
    )
    result = automatic_reference(held)
    assert result.reference_time == 1
    assert result.source_head_top == 275
    all_gaps = (replace(gaps[0], end=Fraction(2)),)
    missing = FaceScaleSeries(
        replace(paired.tops, derived_ranges=all_gaps),
        replace(paired.bottoms, derived_ranges=all_gaps),
    )
    with pytest.raises(ReplayError, match="observed face"):
        automatic_reference(missing)


@pytest.mark.parametrize(
    "config",
    [
        {"headReferencePolicy": "unknown"},
        {"headReferencePolicy": "automatic-face-relative", "referenceTime": "0"},
        {"referenceTime": "0", "referenceHeadTop": float("nan"), "referenceROI": [0, 0, 1, 1]},
        {"referenceTime": "0", "referenceHeadTop": 500, "referenceROI": [0, 0, 0, 1]},
        {"referenceTime": "0", "referenceHeadTop": 500, "referenceROI": [0, 0, 1]},
    ],
)
def test_reference_rejects_inconsistent_settings(
    config: dict[str, object], paired: FaceScaleSeries
) -> None:
    with pytest.raises(ReplayError):
        resolve_reference(config, paired)


def test_explicit_reference_remains_supported(paired: FaceScaleSeries) -> None:
    result = resolve_reference(
        {"referenceTime": "0", "referenceHeadTop": 500, "referenceROI": [0, 900, 400, 400]}, paired
    )
    assert result.source_head_top == 500
    assert result.roi == (0, 900, 400, 400)
