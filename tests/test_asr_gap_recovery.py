import json
import wave
from array import array
from pathlib import Path

import pytest

from replay.asr_gap_recovery import (
    Hole,
    active_threshold,
    frame_levels,
    main,
    merge,
    owned_segments,
    recognized_words,
    recover,
    voiced_holes,
)
from replay.core import ReplayError

RATE = 16000


def voice(seconds: float, loud: list[tuple[float, float]]) -> array[int]:
    samples = array("h")
    for index in range(round(seconds * RATE)):
        moment = index / RATE
        voiced = any(start <= moment < end for start, end in loud)
        samples.append((8000 if voiced else 20) * (1 if index % 2 else -1))
    return samples


def raw_with(words: list[tuple[str, float, float]]) -> dict[str, object]:
    return {
        "language": "ja",
        "text": "".join(text for text, _, _ in words),
        "segments": [
            {
                "start": start,
                "end": end,
                "text": text,
                "words": [{"word": text, "start": start, "end": end, "probability": 0.9}],
            }
            for text, start, end in words
        ],
    }


def test_frame_levels_and_threshold_follow_the_recording_floor() -> None:
    levels = frame_levels(voice(1.0, [(0.0, 0.5)]))
    assert len(levels) == 10
    assert round(levels[0]) == -12
    assert round(levels[9]) == -64
    assert round(active_threshold(levels)) == -45
    assert frame_levels(array("h", [0] * 1600)) == [-120.0]
    with pytest.raises(ReplayError, match="no measurable"):
        active_threshold([])


def test_only_long_voiced_spans_without_words_become_holes() -> None:
    levels = frame_levels(voice(20.0, [(1, 3), (5, 6), (6.5, 8), (12, 13)]))
    words = [(1.2, 2.8), (10.0, 10.5)]
    assert voiced_holes(levels, words) == [Hole(5.0, 8.0, 4.5, 8.5)]
    assert voiced_holes(levels, [(1.2, 2.8), (6.1, 6.3)]) == []


def test_owned_words_are_shifted_and_neighbouring_words_are_dropped() -> None:
    clip = {
        "segments": [
            {
                "avg_logprob": -0.2,
                "compression_ratio": 1.1,
                "no_speech_prob": 0.01,
                "seek": 0,
                "words": [
                    {"word": "前", "start": 0.1, "end": 0.3},
                    {"word": "こんにちは", "start": 1.5, "end": 2.0},
                    {"word": "です", "start": 2.0, "end": 2.4},
                ],
            },
            {"words": [{"word": "後", "start": 4.3, "end": 4.6}]},
        ]
    }
    assert owned_segments(clip, Hole(5.0, 8.0, 4.5, 8.5), offset=4.0) == [
        {
            "avg_logprob": -0.2,
            "compression_ratio": 1.1,
            "no_speech_prob": 0.01,
            "start": 5.5,
            "end": 6.4,
            "text": "こんにちはです",
            "words": [
                {"word": "こんにちは", "start": 5.5, "end": 6.0},
                {"word": "です", "start": 6.0, "end": 6.4},
            ],
            "recovered": True,
        }
    ]
    with pytest.raises(ReplayError, match="malformed"):
        owned_segments(
            {"segments": [{"words": [{"word": 1, "start": 0, "end": 1}]}]},
            Hole(0, 2, 0, 2),
            offset=0,
        )


def test_recovery_inserts_segments_in_order_and_records_every_hole() -> None:
    samples = voice(20.0, [(1, 3), (5, 8), (12, 13)])
    raw = raw_with([("最初", 1.2, 2.8), ("最後", 12.1, 12.9)])
    calls: list[tuple[float, float]] = []

    def transcribe(start: float, end: float) -> object:
        calls.append((start, end))
        return {"segments": [{"words": [{"word": "中間", "start": 2.0, "end": 2.6}]}]}

    completed, provenance = recover(raw, samples, transcribe)
    assert calls == [(3.5, 9.5)]
    assert completed["text"] == "最初中間最後"
    assert json.loads(json.dumps(completed))["segments"][1] == {
        "start": 5.5,
        "end": 6.1,
        "text": "中間",
        "words": [{"word": "中間", "start": 5.5, "end": 6.1}],
        "recovered": True,
    }
    assert provenance["recoveredSegments"] == 1
    assert provenance["holes"] == [
        {
            "start": 5.0,
            "end": 8.0,
            "owner_start": 4.5,
            "owner_end": 8.5,
            "clipStart": 3.5,
            "clipEnd": 9.5,
            "recoveredText": "中間",
        }
    ]


def test_invalid_recognition_shapes_are_refused() -> None:
    with pytest.raises(ReplayError, match="no segment list"):
        recognized_words({"segments": {}})
    with pytest.raises(ReplayError, match="numeric clocks"):
        recognized_words({"segments": [{"words": [{"word": "a", "start": "0", "end": 1}]}]})
    with pytest.raises(ReplayError, match="no segment list"):
        merge({"segments": None}, [])
    assert recognized_words({"segments": [{"words": None}]}) == []


def write_wav(path: Path, samples: array[int], *, rate: int = RATE) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(samples.tobytes())


def test_cli_writes_completed_recognition_and_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_wav(tmp_path / "voice.wav", voice(12.0, [(1, 3), (5, 8)]))
    (tmp_path / "raw.json").write_text(
        json.dumps(raw_with([("最初", 1.2, 2.8)]), ensure_ascii=False), encoding="utf-8"
    )
    seen: list[int] = []

    def transcribe(path: Path) -> object:
        with wave.open(str(path), "rb") as audio:
            seen.append(audio.getnframes())
        return {
            "segments": [
                {"words": [{"word": "続き", "start": 2.0, "end": 2.5, "probability": 0.8}]}
            ]
        }

    arguments = [
        str(tmp_path / "voice.wav"),
        str(tmp_path / "raw.json"),
        str(tmp_path / "out.json"),
        str(tmp_path / "provenance.json"),
    ]
    assert main(arguments, transcribe=transcribe) == 0
    assert json.loads(capsys.readouterr().out) == {"holes": 1, "recovered": 1}
    assert seen == [96000]
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["text"] == "最初続き"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "out.json",
        "provenance.json",
        "raw.json",
        "voice.wav",
    ]
    with pytest.raises(ReplayError, match="overwrite"):
        main(arguments, transcribe=transcribe)


def test_cli_requires_a_recognizer_and_16k_mono_audio(tmp_path: Path) -> None:
    write_wav(tmp_path / "voice.wav", voice(1.0, []), rate=8000)
    (tmp_path / "raw.json").write_text(json.dumps(raw_with([])), encoding="utf-8")
    arguments = [
        str(tmp_path / "voice.wav"),
        str(tmp_path / "raw.json"),
        str(tmp_path / "out.json"),
        str(tmp_path / "provenance.json"),
    ]
    with pytest.raises(ReplayError, match="pinned recognizer"):
        main(arguments)
    with pytest.raises(ReplayError, match="16 kHz mono"):
        main(arguments, transcribe=lambda _path: {})
