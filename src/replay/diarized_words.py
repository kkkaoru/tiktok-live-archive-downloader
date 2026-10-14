"""Align Japanese ASR words to Nemotron speech activity without guessing overlap."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from replay.core import ReplayError
from replay.speaker_subtitles import SpeakerCue


@dataclass(frozen=True)
class TimedWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    speaker: int


def _clock(start: object, end: object) -> tuple[float, float]:
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end <= start
    ):
        raise ReplayError("Invalid speech clock")
    return float(start), float(end)


def read_diarization(path: Path) -> tuple[SpeechSegment, ...]:
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ReplayError("Invalid Nemotron diarization JSON")
    results: list[SpeechSegment] = []
    for record in data["segments"]:
        if not isinstance(record, dict):
            raise ReplayError("Invalid Nemotron speaker segment")
        start, end = _clock(record.get("start"), record.get("end"))
        speaker = record.get("speaker")
        if type(speaker) is not int or not 1 <= speaker <= 8:
            raise ReplayError("Invalid Nemotron speaker index")
        results.append(SpeechSegment(start, end, speaker))
    if not results:
        raise ReplayError("Nemotron detected no speech")
    return tuple(sorted(results, key=lambda item: item.start))


def read_whisper_words(path: Path) -> tuple[TimedWord, ...]:
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ReplayError("Invalid Whisper transcription JSON")
    words: list[TimedWord] = []
    for segment in data["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("words"), list):
            raise ReplayError("Word timestamps are required")
        for word in segment["words"]:
            if not isinstance(word, dict):
                raise ReplayError("Invalid Whisper word")
            start, end = _clock(word.get("start"), word.get("end"))
            text = word.get("word")
            if not isinstance(text, str) or not text.strip():
                raise ReplayError("Invalid Whisper word text")
            words.append(TimedWord(start, end, text))
    if not words:
        raise ReplayError("Whisper returned no timed words")
    return tuple(words)


def align_words(
    words: tuple[TimedWord, ...],
    segments: tuple[SpeechSegment, ...],
    positions: dict[int, str],
) -> tuple[SpeakerCue, ...]:
    """Require a dominant, time-overlapping voice; never infer appearance from order."""
    observed = {segment.speaker for segment in segments}
    if set(positions) != observed or set(positions.values()) != {"left", "middle", "right"}:
        raise ReplayError("Confirm every voice-to-character position before rendering")
    cues: list[SpeakerCue] = []
    for word in words:
        amounts: dict[int, float] = {}
        for segment in segments:
            if segment.start >= word.end:
                break
            overlap = max(0.0, min(word.end, segment.end) - max(word.start, segment.start))
            amounts[segment.speaker] = amounts.get(segment.speaker, 0.0) + overlap
        ranked = sorted(amounts.items(), key=lambda pair: pair[1], reverse=True)
        duration = word.end - word.start
        if (
            not ranked
            or ranked[0][1] < 0.6 * duration
            or (len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.2 * duration)
        ):
            raise ReplayError("Word has no unambiguous speaker; inspect overlapping audio")
        cues.append(SpeakerCue(word.start, word.end, positions[ranked[0][0]], word.text.strip()))
    return tuple(cues)
