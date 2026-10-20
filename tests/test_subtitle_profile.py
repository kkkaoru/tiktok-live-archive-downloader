from dataclasses import replace

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
    japanese = (
        "春の花",
        "遠い海",
        "今日の天気",
        "朝の挨拶",
        "夕焼けの空",
        "白い雲",
        "青い鳥",
        "静かな森",
        "新しい本",
        "明日の予定",
    )
    translations = (
        "spring flowers",
        "distant sea",
        "today's weather",
        "morning greeting",
        "sunset sky",
        "white clouds",
        "blue bird",
        "quiet forest",
        "new book",
        "tomorrow's plans",
    )
    return tuple(
        map(
            lambda index: FrameObservation(
                index,
                index,
                1080,
                1920,
                (
                    TextDetection(japanese[index], 1, TextRectangle(340, 1296, 400, 32)),
                    TextDetection(translations[index], 1, TextRectangle(340, 1408, 400, 24)),
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
        TextDetection("春の花", 1, TextRectangle(340, 1296, 400, 32)),
        TextDetection("spring flowers", 1, TextRectangle(340, 1408, 400, 24)),
    )


def test_changing_counters_do_not_seed_primary_or_secondary_bands(
    observations: tuple[FrameObservation, ...],
) -> None:
    frames = tuple(
        map(
            lambda frame: replace(
                frame,
                detections=(
                    *frame.detections,
                    TextDetection(
                        f"参加者 {frame.actual_time} 人", 1, TextRectangle(340, 288, 400, 32)
                    ),
                    TextDetection(
                        f"count {frame.actual_time}", 1, TextRectangle(340, 1472, 400, 24)
                    ),
                ),
            ),
            observations,
        )
    )
    assert learn_subtitle_bands(frames) == SubtitleBands(1080, 1920, (41,), (44,))
    assert frames[0].detections[-2].text == "参加者 0 人"


def test_fullwidth_counter_changes_alone_refuse_a_blanket_mask() -> None:
    frames = tuple(
        map(
            lambda digit: FrameObservation(
                0,
                0,
                1080,
                1920,
                (TextDetection(f"参加者　{digit}　人", 1, TextRectangle(340, 1296, 400, 32)),),
            ),
            "０１２３４５６７８９",
        )
    )
    with pytest.raises(ValueError, match="blanket mask"):
        learn_subtitle_bands(frames)


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


def test_a_fixed_label_misread_by_ocr_does_not_seed_a_band(
    observations: tuple[FrameObservation, ...],
) -> None:
    readings = (
        "宝箱、ポータル禁止",
        "宝額、ポータル禁止",
        "宝組、ポータル禁止",
        "家稲、ポータル禁止",
        "品額、ポータル禁に",
        "宝箱、ポータル禁止",
        "定箱、ポータル禁止",
        "宝称、ポータル禁止",
        "宝箱、ポータル禁ル",
        "宝縮、ボータル禁止",
    )
    frames = tuple(
        replace(
            frame,
            detections=(
                *frame.detections,
                TextDetection(reading, 1, TextRectangle(388, 364, 592, 80)),
            ),
        )
        for frame, reading in zip(observations, readings, strict=True)
    )
    assert learn_subtitle_bands(frames) == SubtitleBands(1080, 1920, (41,), (44,))


def test_a_label_alone_still_refuses_a_blanket_mask() -> None:
    readings = ("宝箱、ポータル禁止", "宝額、ポータル禁止", "宝組、ポータル禁止") * 4
    frames = tuple(
        FrameObservation(
            0, 0, 1080, 1920, (TextDetection(reading, 1, TextRectangle(388, 364, 592, 80)),)
        )
        for reading in readings
    )
    with pytest.raises(ValueError, match="blanket mask"):
        learn_subtitle_bands(frames)
