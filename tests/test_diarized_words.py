import json
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.diarized_words import (
    SpeechSegment,
    TimedWord,
    align_words,
    read_diarization,
    read_whisper_words,
)
from replay.speaker_subtitles import SpeakerCue


def test_reads_nemotron_and_whisper_json(tmp_path: Path) -> None:
    diarization = tmp_path / "nemo.json"
    diarization.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 2, "end": 3, "speaker": 2},
                    {"start": 0.5, "end": 1.5, "speaker": 1},
                    {"start": 4, "end": 5, "speaker": 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    asr = tmp_path / "whisper.json"
    asr.write_text(
        json.dumps({"segments": [{"words": [{"start": 0.6, "end": 1.3, "word": " こんにちは"}]}]}),
        encoding="utf-8",
    )
    segments = read_diarization(diarization)
    assert segments == (
        SpeechSegment(0.5, 1.5, 1),
        SpeechSegment(2, 3, 2),
        SpeechSegment(4, 5, 3),
    )
    assert read_whisper_words(asr) == (TimedWord(0.6, 1.3, " こんにちは"),)
    assert align_words(
        (TimedWord(0.6, 1.3, " こんにちは"),),
        segments,
        {1: "middle", 2: "left", 3: "right"},
    ) == (SpeakerCue(0.6, 1.3, "middle", "こんにちは"),)


def test_simultaneous_unattributed_word_is_not_assigned() -> None:
    segments = (
        SpeechSegment(0, 2, 1),
        SpeechSegment(0, 2, 2),
        SpeechSegment(3, 4, 3),
    )
    with pytest.raises(ReplayError, match="unambiguous speaker"):
        align_words((TimedWord(0.5, 1.5, "はい"),), segments, {1: "left", 2: "middle", 3: "right"})


def test_missing_speech_or_unknown_position_stops() -> None:
    with pytest.raises(ReplayError, match="unambiguous speaker"):
        align_words(
            (TimedWord(8, 9, "はい"),),
            (SpeechSegment(1, 2, 1), SpeechSegment(3, 4, 2), SpeechSegment(5, 6, 3)),
            {1: "left", 2: "middle", 3: "right"},
        )
    with pytest.raises(ReplayError, match="Confirm every voice"):
        align_words((TimedWord(1, 2, "はい"),), (SpeechSegment(1, 2, 1),), {1: "left"})


@pytest.mark.parametrize(
    ("json_value", "message"),
    [
        ({"segments": []}, "detected no speech"),
        ({"segments": [None]}, "Invalid Nemotron speaker segment"),
        ({"segments": [{"start": 1, "end": 0, "speaker": 1}]}, "Invalid speech clock"),
        ({"segments": [{"start": 0, "end": 1, "speaker": True}]}, "Invalid Nemotron speaker index"),
        ({"segments": [{"start": float("nan"), "end": 1, "speaker": 1}]}, "Invalid speech clock"),
        ({"wrong": 1}, "Invalid Nemotron diarization JSON"),
    ],
)
def test_diarization_validation(tmp_path: Path, json_value: object, message: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(json_value), encoding="utf-8")
    with pytest.raises(ReplayError, match=message):
        read_diarization(path)


@pytest.mark.parametrize(
    ("json_value", "message"),
    [
        ({"wrong": []}, "Invalid Whisper transcription JSON"),
        ({"segments": [None]}, "Word timestamps are required"),
        ({"segments": [{"words": [None]}]}, "Invalid Whisper word"),
        (
            {"segments": [{"words": [{"start": 1, "end": 2, "word": ""}]}]},
            "Invalid Whisper word text",
        ),
        (
            {"segments": [{"words": [{"start": True, "end": 2, "word": "x"}]}]},
            "Invalid speech clock",
        ),
        ({"segments": []}, "returned no timed words"),
    ],
)
def test_whisper_boundary_validation(tmp_path: Path, json_value: object, message: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(json_value), encoding="utf-8")
    with pytest.raises(ReplayError, match=message):
        read_whisper_words(path)
