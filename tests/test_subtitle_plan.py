"""Exact native membership and external mask validation."""

from dataclasses import asdict, replace
from fractions import Fraction

import pytest

from replay.caption_io import FrameObservation
from replay.caption_masks import MaskPolicy, TextDetection, TextRectangle
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_plan import ObservedMask, number, observed_masks, repair_spans
from replay.subtitle_profile import SubtitleBands
from replay.subtitle_stream import SubtitleSpan

PROFILE = SubtitleBands(640, 360, (6,), ())


@pytest.fixture
def observations() -> tuple[FrameObservation, ...]:
    return tuple(
        map(
            lambda index: FrameObservation(
                index,
                index,
                640,
                360,
                (TextDetection(f"字幕{index}", 1, TextRectangle(240, 200, 160, 24)),),
            ),
            range(12),
        )
    )


@pytest.fixture
def policy() -> MaskPolicy:
    return MaskPolicy(search_top=0, search_bottom=1, vertical_line_padding=0, horizontal_padding=0)


def test_observed_masks_use_actual_clocks_and_selected_source_canvas(
    observations: tuple[FrameObservation, ...], policy: MaskPolicy
) -> None:
    frames = (replace(observations[0], requested_time=0.25),) + observations[1:]
    masks = observed_masks(
        tuple(reversed(frames)) + frames, duration=12, policy=policy, profile=PROFILE
    )
    assert masks == (ObservedMask(0, 12, TextRectangle(240, 200, 160, 24)),)
    assert repair_spans(
        [
            {"startSeconds": mask.start, "endSeconds": mask.end, "region": asdict(mask.rectangle)}
            for mask in masks
        ],
        frames=360,
        width=640,
        height=360,
    ) == (SubtitleSpan(0, 360, 240, 200, 160, 24),)


def test_mask_membership_uses_actual_vfr_clocks_not_average_fps() -> None:
    assert repair_spans(
        [
            {
                "startSeconds": 0.01,
                "endSeconds": 0.033,
                "region": {"x": 0, "y": 0, "width": 32, "height": 32},
            }
        ],
        frames=4,
        width=64,
        height=64,
        source_clock=VideoFrameClock((90000, 91170, 92970, 630000), Fraction(1, 90000), 2970),
    ) == (SubtitleSpan(1, 2, 0, 0, 32, 32),)


def test_mask_clock_length_must_match_frame_budget() -> None:
    with pytest.raises(ValueError, match="clock length"):
        repair_spans(
            [],
            frames=2,
            width=64,
            height=64,
            source_clock=VideoFrameClock((0,), Fraction(1, 30), 1),
        )


def test_observed_empty_cell_is_never_filled_by_a_neighbour(
    observations: tuple[FrameObservation, ...], policy: MaskPolicy
) -> None:
    frames = observations[:6] + (replace(observations[6], detections=()),) + observations[7:]
    assert observed_masks(frames, duration=12, policy=policy, profile=PROFILE) == (
        ObservedMask(0, 5.5, TextRectangle(240, 200, 160, 24)),
        ObservedMask(6.5, 12, TextRectangle(240, 200, 160, 24)),
    )


def test_adjacent_observations_cover_transient_ocr_box_truncation(policy: MaskPolicy) -> None:
    frames = (
        FrameObservation(
            0, 0, 640, 360, (TextDetection("字幕全体", 1, TextRectangle(120, 200, 400, 24)),)
        ),
        FrameObservation(
            1, 1, 640, 360, (TextDetection("字幕一部", 1, TextRectangle(240, 200, 160, 24)),)
        ),
        FrameObservation(
            2, 2, 640, 360, (TextDetection("字幕全体", 1, TextRectangle(120, 200, 400, 24)),)
        ),
    )
    assert observed_masks(frames, duration=3, policy=policy, profile=PROFILE) == (
        ObservedMask(0, 3, TextRectangle(120, 200, 400, 24)),
    )


def test_geometry_does_not_leak_across_unobserved_gap(policy: MaskPolicy) -> None:
    frames = (
        FrameObservation(
            0, 0, 640, 360, (TextDetection("左字幕", 1, TextRectangle(20, 200, 100, 24)),)
        ),
        FrameObservation(
            4, 4, 640, 360, (TextDetection("右字幕", 1, TextRectangle(400, 200, 100, 24)),)
        ),
    )
    assert observed_masks(frames, duration=5, policy=policy, profile=PROFILE) == (
        ObservedMask(0, 1, TextRectangle(20, 200, 100, 24)),
        ObservedMask(3, 5, TextRectangle(400, 200, 100, 24)),
    )


def test_long_unobserved_gaps_remain_unmasked(
    observations: tuple[FrameObservation, ...], policy: MaskPolicy
) -> None:
    frames = observations[:10] + (
        replace(observations[10], requested_time=20, actual_time=20),
        replace(observations[11], requested_time=21, actual_time=21),
    )
    assert observed_masks(frames, duration=22, policy=policy, profile=PROFILE) == (
        ObservedMask(0, 10, TextRectangle(240, 200, 160, 24)),
        ObservedMask(19, 22, TextRectangle(240, 200, 160, 24)),
    )


@pytest.mark.parametrize(("duration", "distance"), [(0, 1), (86401, 1), (12, 0), (12, 1.1)])
def test_invalid_observed_mask_bounds(
    observations: tuple[FrameObservation, ...], policy: MaskPolicy, duration: float, distance: float
) -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        observed_masks(
            observations,
            duration=duration,
            maximum_distance=distance,
            policy=policy,
            profile=PROFILE,
        )


def test_empty_observed_mask_input(policy: MaskPolicy) -> None:
    with pytest.raises(ValueError, match="budget"):
        observed_masks((), duration=12, policy=policy, profile=PROFILE)


def test_mixed_observation_canvases_refused(
    observations: tuple[FrameObservation, ...], policy: MaskPolicy
) -> None:
    with pytest.raises(ValueError, match="canvas"):
        observed_masks(
            observations + (replace(observations[0], width=320),),
            duration=12,
            policy=policy,
            profile=PROFILE,
        )


@pytest.mark.parametrize(("actual", "requested"), [(12, 0), (0, 12)])
def test_out_of_source_observation_clock_refused(
    observations: tuple[FrameObservation, ...], policy: MaskPolicy, actual: float, requested: float
) -> None:
    with pytest.raises(ValueError, match="clock"):
        observed_masks(
            observations
            + (replace(observations[0], actual_time=actual, requested_time=requested),),
            duration=12,
            policy=policy,
            profile=PROFILE,
        )


def test_refinement_does_not_relearn_unrelated_centered_text(
    observations: tuple[FrameObservation, ...], policy: MaskPolicy
) -> None:
    refined = tuple(
        map(
            lambda frame: replace(
                frame,
                detections=frame.detections
                + (
                    TextDetection(
                        f"画面表示{frame.requested_time}", 1, TextRectangle(240, 16, 160, 24)
                    ),
                ),
            ),
            observations,
        )
    )
    assert observed_masks(refined, duration=12, policy=policy, profile=PROFILE) == (
        ObservedMask(0, 12, TextRectangle(240, 200, 160, 24)),
    )


def test_selected_profile_must_match_the_source_canvas(
    observations: tuple[FrameObservation, ...], policy: MaskPolicy
) -> None:
    with pytest.raises(ValueError, match="differs"):
        observed_masks(
            observations, duration=12, policy=policy, profile=SubtitleBands(1280, 720, (6,), ())
        )


def test_coalesces_half_open_masks() -> None:
    assert repair_spans(
        [
            {
                "startSeconds": 0,
                "endSeconds": 2 / 30,
                "region": {"x": 8, "y": 8, "width": 16, "height": 16},
            },
            {
                "startSeconds": 2 / 30,
                "endSeconds": 4 / 30,
                "region": {"x": 8, "y": 8, "width": 16, "height": 16},
            },
        ],
        frames=5,
        width=64,
        height=64,
    ) == (SubtitleSpan(0, 4, 8, 8, 16, 16),)


def test_overlaps_union_and_gap_preserved() -> None:
    assert repair_spans(
        [
            {
                "startSeconds": 0,
                "endSeconds": 1 / 30,
                "region": {"x": 0, "y": 0, "width": 16, "height": 16},
            },
            {
                "startSeconds": 0,
                "endSeconds": 1 / 30,
                "region": {"x": 8, "y": 8, "width": 16, "height": 16},
            },
            {
                "startSeconds": 2 / 30,
                "endSeconds": 3 / 30,
                "region": {"x": 0, "y": 0, "width": 16, "height": 16},
            },
        ],
        frames=4,
        width=64,
        height=64,
    ) == (SubtitleSpan(0, 1, 0, 0, 24, 24), SubtitleSpan(2, 3, 0, 0, 16, 16))


@pytest.mark.parametrize("value", [None, True, "1", float("nan")])
def test_invalid_number(value: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        number(value)


def test_invalid_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        repair_spans([], frames=0, width=64, height=64)


def test_invalid_list() -> None:
    with pytest.raises(ValueError, match="list"):
        repair_spans(None, frames=4, width=64, height=64)


def test_invalid_object() -> None:
    with pytest.raises(ValueError, match="object"):
        repair_spans([{}], frames=4, width=64, height=64)


def test_nonintegral_pixels_refused() -> None:
    with pytest.raises(ValueError, match="Nonintegral"):
        repair_spans(
            [
                {
                    "startSeconds": 0,
                    "endSeconds": 1,
                    "region": {"x": 0.5, "y": 0, "width": 16, "height": 16},
                }
            ],
            frames=4,
            width=64,
            height=64,
        )


def test_out_of_canvas_refused() -> None:
    with pytest.raises(ValueError, match="bounds"):
        repair_spans(
            [
                {
                    "startSeconds": 0,
                    "endSeconds": 1,
                    "region": {"x": 60, "y": 0, "width": 16, "height": 16},
                }
            ],
            frames=4,
            width=64,
            height=64,
        )
