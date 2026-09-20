import pytest

from replay.caption_io import FrameObservation
from replay.caption_masks import TextDetection, TextRectangle
from replay.subtitle_profile import (
    SubtitleBands,
    japanese_letters,
    learn_subtitle_bands,
    profiled_detections,
)


@pytest.fixture
def observations() -> tuple[FrameObservation, ...]:
    return tuple(
        map(
            lambda index: FrameObservation(
                index,
                index,
                1080,
                1920,
                (
                    TextDetection(f"字幕{index}", 1, TextRectangle(340, 1296, 400, 32)),
                    TextDetection(f"translation{index}", 1, TextRectangle(340, 1408, 400, 24)),
                    TextDetection("STATIC", 1, TextRectangle(400, 1200, 300, 24)),
                    TextDetection(f"shirt{index}", 1, TextRectangle(440, 1220, 200, 24)),
                ),
            ),
            range(10),
        )
    )


def test_variable_japanese_and_lower_translation_seed_bands(
    observations: tuple[FrameObservation, ...],
) -> None:
    assert learn_subtitle_bands(observations) == SubtitleBands(1080, 1920, (41,), (44,))


def test_static_and_upper_roman_text_do_not_seed_a_blanket_mask(
    observations: tuple[FrameObservation, ...],
) -> None:
    result = profiled_detections(observations[0], learn_subtitle_bands(observations))
    assert result == (
        TextDetection("字幕0", 1, TextRectangle(340, 1296, 400, 32)),
        TextDetection("translation0", 1, TextRectangle(340, 1408, 400, 24)),
    )


def test_isolated_upper_object_rejected_but_supported_wrapped_line_kept() -> None:
    frame = FrameObservation(
        0,
        0,
        1080,
        1920,
        (
            TextDetection("字幕", 1, TextRectangle(340, 1296, 400, 32)),
            TextDetection("続き", 1, TextRectangle(480, 1272, 120, 32)),
            TextDetection("badge", 1, TextRectangle(760, 1272, 80, 24)),
        ),
    )
    assert profiled_detections(frame, SubtitleBands(1080, 1920, (41,), ())) == (
        TextDetection("字幕", 1, TextRectangle(340, 1296, 400, 32)),
        TextDetection("続き", 1, TextRectangle(480, 1272, 120, 32)),
    )


def test_empty_and_low_confidence_are_not_invented_detections() -> None:
    frame = FrameObservation(
        0, 0, 1080, 1920, (TextDetection("字幕", 0.1, TextRectangle(340, 1296, 400, 32)),)
    )
    assert profiled_detections(frame, SubtitleBands(1080, 1920, (41,), ())) == ()


def test_script_detection_does_not_treat_roman_or_long_vowel_mark_as_japanese() -> None:
    assert japanese_letters("語句") is True
    assert japanese_letters("KANAー") is False


def test_empty_profile_input() -> None:
    with pytest.raises(ValueError, match="budget"):
        learn_subtitle_bands(())


def test_static_japanese_alone_is_not_enough() -> None:
    frames = (
        FrameObservation(
            0, 0, 1080, 1920, (TextDetection("固定", 1, TextRectangle(340, 1296, 400, 32)),)
        ),
    ) * 10
    with pytest.raises(ValueError, match="blanket mask"):
        learn_subtitle_bands(frames)


def test_wrong_canvas() -> None:
    with pytest.raises(ValueError, match="canvas"):
        learn_subtitle_bands((FrameObservation(0, 0, 0, 1920, ()),))


def test_inconsistent_frames() -> None:
    with pytest.raises(ValueError, match="Inconsistent"):
        learn_subtitle_bands(
            (FrameObservation(0, 0, 1080, 1920, ()), FrameObservation(1, 1, 1920, 1080, ()))
        )


def test_outside_canvas_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds canvas"):
        learn_subtitle_bands(
            (
                FrameObservation(
                    0,
                    0,
                    1080,
                    1920,
                    (TextDetection("字幕", 1, TextRectangle(1000, 1300, 200, 30)),),
                ),
            )
        )


def test_frame_profile_mismatch() -> None:
    with pytest.raises(ValueError, match="differs"):
        profiled_detections(
            FrameObservation(0, 0, 1920, 1080, ()), SubtitleBands(1080, 1920, (41,), ())
        )


def test_missing_bands_rejected() -> None:
    with pytest.raises(ValueError, match="Empty"):
        profiled_detections(
            FrameObservation(0, 0, 1080, 1920, ()), SubtitleBands(1080, 1920, (), ())
        )
