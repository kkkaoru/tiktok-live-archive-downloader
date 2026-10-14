"""Offline synthetic audits; no claims of real footage or temporal approval."""

from dataclasses import replace

import numpy as np
import pytest

from replay.caption_io import FrameObservation
from replay.caption_masks import TextDetection, TextRectangle
from replay.subtitle_audit import (
    MaskObservationAudit,
    OutputTextAudit,
    PixelSupportAudit,
    audit_mask_observations,
    audit_output_text,
    audit_pixel_support,
)
from replay.subtitle_plan import ObservedMask
from replay.subtitle_profile import SubtitleBands


def test_pixel_audit_separates_roi_leaks_and_persistent_support() -> None:
    result = audit_pixel_support(
        np.array([[True, True, False], [False, False, True]]),
        np.array([[True, False, True], [False, False, True]]),
        allowed_roi=np.array([[True, True, False], [False, False, False]]),
    )
    assert result == PixelSupportAudit(3, 3, 2, 1, 2)


def test_empty_pixel_support_is_zero_not_quality_approval() -> None:
    result = audit_pixel_support(
        np.zeros((2, 3), dtype=np.bool_),
        np.zeros((2, 3), dtype=np.bool_),
        allowed_roi=np.ones((2, 3), dtype=np.bool_),
    )
    assert result == PixelSupportAudit(0, 0, 0, 0, 0)


def test_mismatched_pixel_support_is_refused() -> None:
    with pytest.raises(ValueError, match="pixel-support"):
        audit_pixel_support(
            np.zeros((2, 3), dtype=np.bool_),
            np.zeros((3, 2), dtype=np.bool_),
            allowed_roi=np.zeros((2, 3), dtype=np.bool_),
        )


def test_non_image_pixel_support_is_refused() -> None:
    with pytest.raises(ValueError, match="pixel-support"):
        audit_pixel_support(
            np.zeros((0, 0), dtype=np.bool_),
            np.zeros((0, 0), dtype=np.bool_),
            allowed_roi=np.zeros((0, 0), dtype=np.bool_),
        )


@pytest.fixture
def frame() -> FrameObservation:
    return FrameObservation(
        1, 1, 640, 360, (TextDetection("合成字幕", 1, TextRectangle(240, 200, 160, 24)),)
    )


@pytest.fixture
def profile() -> SubtitleBands:
    return SubtitleBands(640, 360, (6,), ())


def test_mask_audit_checks_actual_time_and_half_open_end(
    frame: FrameObservation, profile: SubtitleBands
) -> None:
    frames = (frame, replace(frame, requested_time=2.25, actual_time=2))
    masks = (ObservedMask(0, 2, TextRectangle(240, 200, 160, 24)),)
    assert audit_mask_observations(frames, masks, profile=profile) == MaskObservationAudit(2, 2, 1)


@pytest.mark.parametrize(
    "box",
    [
        TextRectangle(241, 200, 159, 24),
        TextRectangle(240, 201, 160, 23),
        TextRectangle(240, 200, 159, 24),
        TextRectangle(240, 200, 160, 23),
    ],
)
def test_any_uncovered_line_edge_is_reported(
    frame: FrameObservation, profile: SubtitleBands, box: TextRectangle
) -> None:
    assert audit_mask_observations(
        (frame,), (ObservedMask(0, 2, box),), profile=profile
    ) == MaskObservationAudit(1, 1, 1)


def test_missing_masks_are_not_silently_accepted(
    frame: FrameObservation, profile: SubtitleBands
) -> None:
    assert audit_mask_observations((frame,), (), profile=profile) == MaskObservationAudit(1, 1, 1)


@pytest.mark.parametrize(
    "masks",
    [
        (ObservedMask(0, 0, TextRectangle(0, 0, 10, 10)),),
        (
            ObservedMask(0, 2, TextRectangle(0, 0, 10, 10)),
            ObservedMask(1, 3, TextRectangle(0, 0, 10, 10)),
        ),
        (ObservedMask(0, 2, TextRectangle(630, 200, 20, 24)),),
    ],
)
def test_invalid_masks_refused(
    frame: FrameObservation, profile: SubtitleBands, masks: tuple[ObservedMask, ...]
) -> None:
    with pytest.raises(ValueError):
        audit_mask_observations((frame,), masks, profile=profile)


def test_missing_and_invalid_source_observations_refused(
    frame: FrameObservation, profile: SubtitleBands
) -> None:
    with pytest.raises(ValueError, match="budget"):
        audit_mask_observations((), (), profile=profile)
    with pytest.raises(ValueError, match="clock"):
        audit_mask_observations((replace(frame, actual_time=-1),), (), profile=profile)


def test_output_audit_reports_readable_residue_and_retains_actual_clock_difference(
    frame: FrameObservation, profile: SubtitleBands
) -> None:
    assert audit_output_text(
        (frame,), (replace(frame, actual_time=0.75),), profile=profile
    ) == OutputTextAudit(1, 1, 1, 1, 0.25)
    assert audit_output_text(
        (frame,), (replace(frame, detections=()),), profile=profile
    ) == OutputTextAudit(1, 1, 0, 0, 0)


def test_output_title_outside_source_bands_is_not_old_subtitle_residue(
    frame: FrameObservation, profile: SubtitleBands
) -> None:
    output = replace(
        frame, detections=(TextDetection("NEW TITLE", 1, TextRectangle(240, 32, 160, 24)),)
    )
    assert audit_output_text((frame,), (output,), profile=profile) == OutputTextAudit(1, 1, 0, 0, 0)


def test_missing_duplicated_or_mismatched_requests_refused(
    frame: FrameObservation, profile: SubtitleBands
) -> None:
    with pytest.raises(ValueError, match="Incomplete"):
        audit_output_text((frame,), (), profile=profile)
    with pytest.raises(ValueError, match="requests"):
        audit_output_text((frame, frame), (frame, frame), profile=profile)
    with pytest.raises(ValueError, match="requests"):
        audit_output_text((frame,), (replace(frame, requested_time=2),), profile=profile)


@pytest.mark.parametrize(("requested", "actual"), [(-0.5, 1), (1, -1)])
def test_invalid_paired_clocks_refused(
    frame: FrameObservation, profile: SubtitleBands, requested: float, actual: float
) -> None:
    invalid = replace(frame, requested_time=requested, actual_time=actual)
    with pytest.raises(ValueError):
        audit_output_text((invalid,), (invalid,), profile=profile)


def test_candidate_canvas_must_match_source(
    frame: FrameObservation, profile: SubtitleBands
) -> None:
    with pytest.raises(ValueError, match="differs"):
        audit_output_text((frame,), (replace(frame, width=1280),), profile=profile)
