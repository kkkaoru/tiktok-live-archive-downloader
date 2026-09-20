import math

import pytest

from replay import caption_io
from replay.caption_io import MLXWordGroup, mlx_groups, ocr_frames, speech_tokens
from replay.caption_masks import TextDetection, TextRectangle
from replay.caption_timing import SpokenToken


def response(measurement: object) -> dict[str, object]:
    return {
        "ok": True,
        "data": {"structuredContent": {"sourceModified": False, "measurement": measurement}},
    }


def speech(segments: object, duration: object = 2) -> dict[str, object]:
    return {
        "onDevice": True,
        "humanReviewed": False,
        "durationSeconds": duration,
        "segments": segments,
    }


def test_native_speech_is_parsed_without_implicit_cloud_or_timing_fallback() -> None:
    assert speech_tokens(
        response(speech([{"text": "こんにちは。", "startSeconds": 0.2, "durationSeconds": 1}])),
        utterance="one",
    ) == (SpokenToken("こんにちは。", 0.2, 1.2, "one"),)
    assert speech_tokens(response(speech([])), utterance="empty") == ()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {1: "bad"},
        {"ok": False},
        {"ok": True, "data": {"isError": True}},
        {"ok": True, "data": {"structuredContent": {"sourceModified": True}}},
        dict.fromkeys(map(str, range(129))),
    ],
)
def test_executor_failures_or_malformed_envelopes_are_not_measurements(payload: object) -> None:
    with pytest.raises(ValueError):
        speech_tokens(payload, utterance="one")


@pytest.mark.parametrize(
    "measurement",
    [
        {"onDevice": False},
        {"onDevice": True, "humanReviewed": True},
        speech([], True),
        speech([], "2"),
        speech([], math.nan),
        speech([], 10**400),
        speech([], 61),
        speech({}, 2),
        speech([{}] * 1001),
        speech([{"text": "範囲外", "startSeconds": 1, "durationSeconds": 2}]),
        speech([{"text": 1, "startSeconds": 0, "durationSeconds": 1}]),
        speech([{"text": "あ" * 2048, "startSeconds": 0, "durationSeconds": 1}] * 6),
    ],
)
def test_speech_boundary_types_and_budgets_are_validated(measurement: object) -> None:
    with pytest.raises(ValueError):
        speech_tokens(response(measurement), utterance="one")


def test_ocr_retains_actual_and_requested_times_and_global_coordinates() -> None:
    result = ocr_frames(
        response(
            [
                {
                    "width": 1080,
                    "height": 1920,
                    "requestedTimeSeconds": 15,
                    "actualTimeSeconds": 14,
                    "lines": [
                        {
                            "text": "合成字幕",
                            "confidence": 0.8,
                            "region": {"x": 20, "y": 1260, "width": 500, "height": 30},
                        }
                    ],
                }
            ]
        )
    )
    assert len(result) == 1
    assert result[0].requested_time == 15
    assert result[0].actual_time == 14
    assert result[0].width == 1080
    assert result[0].height == 1920
    assert result[0].detections == (
        TextDetection("合成字幕", 0.8, TextRectangle(20, 1260, 500, 30)),
    )
    assert ocr_frames(response([])) == ()


@pytest.mark.parametrize(
    ("width", "height", "requested", "actual"),
    [(100.1, 100, 0, 0), (8192, 8192, 0, 0), (100, 100, -1, 0), (100, 100, 0, -1)],
)
def test_invalid_ocr_canvas_and_times_are_rejected(
    width: float, height: float, requested: float, actual: float
) -> None:
    with pytest.raises(ValueError):
        ocr_frames(
            response(
                [
                    {
                        "width": width,
                        "height": height,
                        "requestedTimeSeconds": requested,
                        "actualTimeSeconds": actual,
                        "lines": [],
                    }
                ]
            )
        )


def test_ocr_batch_text_budget_is_not_truncated() -> None:
    with pytest.raises(ValueError, match="batch text budget"):
        ocr_frames(
            response(
                [
                    {
                        "width": 100,
                        "height": 100,
                        "requestedTimeSeconds": 0,
                        "actualTimeSeconds": 0,
                        "lines": [
                            {
                                "text": "あ" * 8000,
                                "confidence": 1,
                                "region": {"x": 0, "y": 50, "width": 10, "height": 10},
                            }
                        ]
                        * 2,
                    }
                ]
            )
        )


def test_mlx_coincident_pieces_keep_native_onsets_without_interpolation() -> None:
    groups = mlx_groups(
        {
            "language": "ja",
            "text": "ジョの",
            "segments": [
                {
                    "start": 1.4,
                    "end": 1.58,
                    "text": "ジョの",
                    "words": [
                        {"word": "ジ", "start": 1.4, "end": 1.46, "probability": 0.9},
                        {"word": "ョ", "start": 1.46, "end": 1.46, "probability": 0.8},
                        {"word": "の", "start": 1.46, "end": 1.58, "probability": 0.7},
                    ],
                }
            ],
        },
        duration=2,
    )
    assert groups == (
        MLXWordGroup(
            text="ジ",
            start=1.4,
            end=1.46,
            source_segments=(0,),
            piece_count=1,
            minimum_probability=0.9,
        ),
        MLXWordGroup(
            text="ョの",
            start=1.46,
            end=1.58,
            source_segments=(0,),
            piece_count=2,
            minimum_probability=0.7,
        ),
    )
    assert groups[1].spoken_token() == SpokenToken("ョの", 1.46, 1.58, "mlx:0")


def test_mlx_coincident_segment_boundary_resets_utterance_and_keeps_provenance() -> None:
    assert mlx_groups(
        {
            "language": "ja",
            "text": " 合成。 ",
            "segments": [
                {
                    "start": 0,
                    "end": 0,
                    "text": "合",
                    "words": [{"word": "合", "start": 0, "end": 0, "probability": 0.5}],
                },
                {
                    "start": 0,
                    "end": 1,
                    "text": "成。",
                    "words": [{"word": "成。", "start": 0, "end": 1, "probability": 0.8}],
                },
            ],
        },
        duration=2,
    ) == (
        MLXWordGroup(
            text="合成。",
            start=0,
            end=1,
            source_segments=(0, 1),
            piece_count=2,
            minimum_probability=0.5,
        ),
    )


def test_mlx_isolated_points_remain_explicit_and_cannot_become_spoken_intervals() -> None:
    groups = mlx_groups(
        {
            "language": "ja",
            "text": "次",
            "segments": [
                {
                    "start": 1,
                    "end": 1,
                    "text": "次",
                    "words": [{"word": "次", "start": 1, "end": 1, "probability": 0.7}],
                }
            ],
        },
        duration=2,
    )
    assert groups == (
        MLXWordGroup(
            text="次",
            start=1,
            end=1,
            source_segments=(0,),
            piece_count=1,
            minimum_probability=0.7,
        ),
    )
    with pytest.raises(ValueError, match="speech token interval"):
        groups[0].spoken_token()
    assert mlx_groups({"language": "ja", "text": "", "segments": []}, duration=1) == ()


@pytest.mark.parametrize("duration", [0, 86401, math.nan, True])
def test_mlx_requires_a_bounded_external_audio_clock(duration: float) -> None:
    with pytest.raises(ValueError):
        mlx_groups({"language": "ja", "text": "", "segments": []}, duration=duration)


@pytest.mark.parametrize(
    ("text", "start", "end", "probability", "segments", "count"),
    [
        (" ", 0, 1, 0.5, (0,), 1),
        ("\0", 0, 1, 0.5, (0,), 1),
        ("あ" * 2049, 0, 1, 0.5, (0,), 1),
        ("次", math.nan, 1, 0.5, (0,), 1),
        ("次", 0, math.inf, 0.5, (0,), 1),
        ("次", -1, 1, 0.5, (0,), 1),
        ("次", 0, -1, 0.5, (0,), 1),
        ("次", 0, 1, -0.1, (0,), 1),
        ("次", 0, 1, 1.1, (0,), 1),
        ("次", 0, 1, 0.5, (0,), 0),
        ("次", 0, 1, 0.5, (), 1),
        ("次", 0, 1, 0.5, (-1,), 1),
    ],
)
def test_mlx_group_validates_owned_records(
    text: str,
    start: float,
    end: float,
    probability: float,
    segments: tuple[int, ...],
    count: int,
) -> None:
    with pytest.raises(ValueError):
        MLXWordGroup(
            text=text,
            start=start,
            end=end,
            source_segments=segments,
            piece_count=count,
            minimum_probability=probability,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"language": "en", "text": "", "segments": []},
        {"language": "ja", "text": "残す", "segments": []},
        {
            "language": "ja",
            "text": "残す",
            "segments": [{"start": 0, "end": 1, "text": "残す", "words": []}],
        },
        {
            "language": "ja",
            "text": "",
            "segments": [{"start": 0, "end": 3, "text": "", "words": []}],
        },
        {
            "language": "ja",
            "text": "次",
            "segments": [
                {
                    "start": 1,
                    "end": 2,
                    "text": "次",
                    "words": [{"word": "次", "start": 0, "end": 3, "probability": 0.8}],
                }
            ],
        },
        {
            "language": "ja",
            "text": "後先",
            "segments": [
                {
                    "start": 0,
                    "end": 2,
                    "text": "後先",
                    "words": [
                        {"word": "後", "start": 1, "end": 2, "probability": 0.8},
                        {"word": "先", "start": 0, "end": 1, "probability": 0.8},
                    ],
                }
            ],
        },
    ],
)
def test_mlx_rejects_unordered_clocks_out_of_scope_words_and_lost_text(payload: object) -> None:
    with pytest.raises(ValueError):
        mlx_groups(payload, duration=2)


def test_mlx_preserves_finer_word_clocks_without_clamping_to_coarse_segments() -> None:
    groups = mlx_groups(
        {
            "language": "ja",
            "text": "次",
            "segments": [
                {
                    "start": 1,
                    "end": 1.1,
                    "text": "次",
                    "words": [{"word": "次", "start": 0.9, "end": 1.2, "probability": 0.8}],
                }
            ],
        },
        duration=2,
    )
    assert groups[0].spoken_token() == SpokenToken("次", 0.9, 1.2, "mlx:0")


def test_mlx_text_budget_is_not_truncated() -> None:
    with pytest.raises(ValueError, match="text budget"):
        mlx_groups({"language": "ja", "text": "あ" * 1333334, "segments": []}, duration=1)


@pytest.mark.parametrize(("setting", "limit"), [("MAX_MLX_WORDS", 1), ("MAX_MLX_TEXT_BYTES", 4)])
def test_mlx_total_word_budgets_apply_across_segments(
    monkeypatch: pytest.MonkeyPatch, setting: str, limit: int
) -> None:
    monkeypatch.setattr(caption_io, setting, limit)
    with pytest.raises(ValueError, match="word budget"):
        mlx_groups(
            {
                "language": "ja",
                "text": "あ",
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": "あ",
                        "words": [{"word": "あ", "start": 0, "end": 1, "probability": 0.9}],
                    },
                    {
                        "start": 1,
                        "end": 2,
                        "text": "あ",
                        "words": [{"word": "あ", "start": 1, "end": 2, "probability": 0.9}],
                    },
                ],
            },
            duration=2,
        )
