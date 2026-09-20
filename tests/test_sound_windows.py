"""Bound local SoundAnalysis results without treating scores as quiet evidence."""

from copy import deepcopy

import pytest

from replay.caption_io import SoundWindow, sound_windows


@pytest.fixture
def measurement() -> dict[str, object]:
    return {
        "durationSeconds": 1,
        "humanReviewed": False,
        "windowDurationSeconds": 0.5,
        "overlapFactor": 0.5,
        "windows": [
            {
                "startSeconds": 0,
                "durationSeconds": 0.5,
                "speechConfidence": 0.2,
                "musicConfidence": 0.9,
            },
            {
                "startSeconds": 0.25,
                "durationSeconds": 0.5,
                "speechConfidence": 0.7,
                "musicConfidence": 0.4,
            },
        ],
    }


def envelope(measurement: object) -> dict[str, object]:
    return {
        "ok": True,
        "data": {"structuredContent": {"sourceModified": False, "measurement": measurement}},
    }


def test_raw_overlapping_scores_are_preserved(measurement: dict[str, object]) -> None:
    result = sound_windows(envelope(measurement))
    assert result == (SoundWindow(0, 0.5, 0.2, 0.9), SoundWindow(0.25, 0.75, 0.7, 0.4))


@pytest.mark.parametrize(
    "changes",
    [
        {"durationSeconds": 0.1},
        {"durationSeconds": 61},
        {"humanReviewed": True},
        {"windowDurationSeconds": 1},
        {"overlapFactor": 0.25},
    ],
)
def test_unexpected_policy(measurement: dict[str, object], changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="policy"):
        sound_windows(envelope(measurement | changes))


def test_empty_result_is_not_silence(measurement: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="cannot infer silence"):
        sound_windows(envelope(measurement | {"windows": []}))


def test_window_exceeds_actual_input(measurement: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="inside"):
        sound_windows(envelope(measurement | {"durationSeconds": 0.5}))


def test_reversed_windows_rejected(measurement: dict[str, object]) -> None:
    changed = deepcopy(measurement)
    rows = changed["windows"]
    assert isinstance(rows, list)
    changed["windows"] = list(reversed(rows))
    with pytest.raises(ValueError, match="ordered"):
        sound_windows(envelope(changed))


@pytest.mark.parametrize("start,end", [(-1, 0.5), (0.5, 0.5), (0, 61)])
def test_invalid_clock(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="clock"):
        SoundWindow(start, end, 0.5, 0.5)


@pytest.mark.parametrize("speech,music", [(-0.1, 0), (1.1, 0), (0, -0.1), (0, 1.1)])
def test_invalid_confidence(speech: float, music: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        SoundWindow(0, 0.5, speech, music)


def test_nonfinite_score() -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        SoundWindow(0, 0.5, float("nan"), 0.5)
