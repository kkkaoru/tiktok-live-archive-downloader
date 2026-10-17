"""The audit flags confident residual lines inside the search region and names the interval."""

import pytest

from replay import subtitle_repair_audit as audit
from replay.caption_io import FrameObservation
from replay.caption_masks import TextDetection, TextRectangle

REGION = (0, 100, 1000, 300)


def _frame(*lines: TextDetection) -> FrameObservation:
    return FrameObservation(0.0, 0.0, 1080, 1920, lines)


def test_audit_indices_include_strides_and_the_last_frame() -> None:
    assert audit.audit_indices(25, 10) == (0, 10, 20, 24)


def test_audit_indices_reject_bad_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        audit.audit_indices(0)


def test_a_confident_line_in_the_region_is_a_residual() -> None:
    real = TextDetection("決着", 1.0, TextRectangle(100, 150, 60, 30))
    found = audit.residual_lines({3: _frame(real)}, [None, None, None, REGION])
    assert found == (audit.Residual(3, "決着", 1.0, (100.0, 150.0, 60.0, 30.0)),)


def test_low_confidence_smear_guesses_are_ignored() -> None:
    guesses = (
        TextDetection("Ter.ne", 0.3, TextRectangle(100, 150, 60, 30)),
        TextDetection("a6", 0.5, TextRectangle(300, 150, 60, 30)),
    )
    assert audit.residual_lines({0: _frame(*guesses)}, [REGION]) == ()


def test_confident_lines_outside_the_region_or_without_a_region_are_ignored() -> None:
    outside = TextDetection("849/5000", 1.0, TextRectangle(100, 900, 60, 30))
    inside = TextDetection("決着", 1.0, TextRectangle(100, 150, 60, 30))
    observed = {0: _frame(outside), 1: _frame(inside)}
    assert audit.residual_lines(observed, [REGION, None]) == ()


def test_escalation_frames_cover_the_intervals_around_failures() -> None:
    residuals = [audit.Residual(20, "x", 1.0)]
    assert audit.escalation_frames(residuals, 25, 5) == (16, 17, 18, 19, 20, 21, 22, 23, 24)
    assert audit.escalation_frames([], 25, 5) == ()
