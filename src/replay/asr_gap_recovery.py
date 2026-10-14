"""Re-recognize voiced spans that a long-form recognition pass skipped.

Long-form Whisper decoding can skip a whole 30-second window. Spans where the
separated voice is clearly audible but no word was recognized are re-recognized
as short clips with the same pinned model and settings. Only words lying wholly
inside the hole, between the neighbouring recognized words, are added, so
existing clocks are never changed and no clock overlaps or reorders them. Every
hole and recovered segment is recorded as provenance; audio level alone is never
turned into text.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import wave
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from replay.caption_io import mlx_groups
from replay.core import ReplayError
from replay.recordings import record

FRAME_SECONDS = 0.1
FLOOR_PERCENTILE = 0.2
ACTIVE_ABOVE_FLOOR_DB = 18.0
ACTIVE_MINIMUM_DB = -45.0
SILENT_DB = -120.0
# Voiced runs separated by less than this, with no recognized word between them,
# belong to one hole.
MERGE_GAP_SECONDS = 3.0
MINIMUM_HOLE_SECONDS = 2.0
OWNERSHIP_PADDING_SECONDS = 0.5
CLIP_PADDING_SECONDS = 1.0
SAMPLE_RATE = 16000
MAXIMUM_HOLES = 5000
# The recognizer's own scores for the clip segment travel with its owned words, so
# downstream audits judge a recovered segment exactly like a full-pass segment.
SEGMENT_SCORES = ("avg_logprob", "compression_ratio", "no_speech_prob", "temperature")

Transcriber = Callable[[Path], object]


@dataclass(frozen=True, slots=True)
class Hole:
    """A voiced span without recognized words, and the interval it may own."""

    start: float
    end: float
    owner_start: float
    owner_end: float


def frame_levels(samples: array[int], *, rate: int = SAMPLE_RATE) -> list[float]:
    """RMS level in dBFS for each 100 ms frame of signed 16-bit mono samples."""
    size = round(rate * FRAME_SECONDS)
    levels: list[float] = []
    for first in range(0, len(samples) - size + 1, size):
        frame = samples[first : first + size]
        power = sum(value * value for value in frame) / size
        levels.append(10 * math.log10(power / 32768**2) if power else SILENT_DB)
    return levels


def active_threshold(levels: list[float]) -> float:
    """Clearly audible voice: well above this recording's own noise floor."""
    if not levels:
        raise ReplayError("Voice audio has no measurable frames")
    floor = sorted(levels)[int(len(levels) * FLOOR_PERCENTILE)]
    return max(floor + ACTIVE_ABOVE_FLOOR_DB, ACTIVE_MINIMUM_DB)


def recognized_words(raw: object) -> list[tuple[float, float]]:
    words: list[tuple[float, float]] = []
    segments = record(raw).get("segments")
    if not isinstance(segments, list):
        raise ReplayError("Recognition has no segment list")
    for segment in segments:
        items = record(segment).get("words")
        for item in items if isinstance(items, list) else []:
            word = record(item)
            start, end = word.get("start"), word.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ReplayError("Recognized word lacks numeric clocks")
            words.append((float(start), float(end)))
    return sorted(words)


def _voiced_runs(levels: list[float]) -> list[tuple[float, float]]:
    threshold = active_threshold(levels)
    runs: list[tuple[float, float]] = []
    start: int | None = None
    for index, level in enumerate([*levels, SILENT_DB]):
        if level > threshold and start is None:
            start = index
        elif level <= threshold and start is not None:
            runs.append((start * FRAME_SECONDS, index * FRAME_SECONDS))
            start = None
    return runs


def _has_word(words: list[tuple[float, float]], start: float, end: float) -> bool:
    return any(word_start < end and word_end > start for word_start, word_end in words)


def voiced_holes(levels: list[float], words: list[tuple[float, float]]) -> list[Hole]:
    """Voiced spans of at least two seconds that no recognized word overlaps."""
    runs = [run for run in _voiced_runs(levels) if not _has_word(words, *run)]
    merged: list[tuple[float, float]] = []
    for start, end in runs:
        if (
            merged
            and start - merged[-1][1] < MERGE_GAP_SECONDS
            and not _has_word(words, merged[-1][1], start)
        ):
            start = merged.pop()[0]
        merged.append((start, end))
    duration = len(levels) * FRAME_SECONDS
    holes: list[Hole] = []
    for start, end in merged:
        if end - start < MINIMUM_HOLE_SECONDS:
            continue
        before = max((word_end for _, word_end in words if word_end <= start), default=0.0)
        after = min((word_start for word_start, _ in words if word_start >= end), default=duration)
        holes.append(
            Hole(
                start,
                end,
                max(start - OWNERSHIP_PADDING_SECONDS, before),
                min(end + OWNERSHIP_PADDING_SECONDS, after),
            )
        )
    if len(holes) > MAXIMUM_HOLES:
        raise ReplayError("Recognition hole budget exceeded")
    return holes


def owned_segments(clip: object, hole: Hole, *, offset: float) -> list[dict[str, object]]:
    """Shift clip segments to the source clock and keep only words within the hole."""
    result: list[dict[str, object]] = []
    segments = record(clip).get("segments")
    for segment in segments if isinstance(segments, list) else []:
        kept: list[dict[str, object]] = []
        scores = {key: record(segment)[key] for key in SEGMENT_SCORES if key in record(segment)}
        items = record(segment).get("words")
        for item in items if isinstance(items, list) else []:
            word = record(item)
            text, start, end = word.get("word"), word.get("start"), word.get("end")
            if (
                not isinstance(text, str)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
            ):
                raise ReplayError("Recovered word is malformed")
            start, end = float(start) + offset, float(end) + offset
            if hole.owner_start <= start and end <= hole.owner_end:
                kept.append({**word, "start": start, "end": end})
        if kept:
            first, last = kept[0]["start"], kept[-1]["end"]
            result.append(
                {
                    **scores,
                    "start": first,
                    "end": last,
                    "text": "".join(str(word["word"]) for word in kept),
                    "words": kept,
                    "recovered": True,
                }
            )
    return result


def merge(raw: object, recovered: list[dict[str, object]]) -> dict[str, object]:
    """Insert recovered segments in clock order and rebuild the conserved text."""
    data = dict(record(raw))
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ReplayError("Recognition has no segment list")
    combined = sorted(
        [record(segment) for segment in segments] + recovered,
        key=_segment_start,
    )
    data["segments"] = combined
    data["text"] = "".join(str(segment.get("text", "")) for segment in combined)
    return data


def _segment_start(segment: dict[str, object]) -> float:
    start = segment.get("start")
    return float(start) if isinstance(start, (int, float)) else 0.0


def recover(
    raw: object, samples: array[int], transcribe: Callable[[float, float], object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the completed recognition and a provenance record of every hole."""
    levels = frame_levels(samples)
    holes = voiced_holes(levels, recognized_words(raw))
    duration = len(samples) / SAMPLE_RATE
    recovered: list[dict[str, object]] = []
    report: list[dict[str, object]] = []
    for hole in holes:
        clip_start = max(0.0, hole.owner_start - CLIP_PADDING_SECONDS)
        clip_end = min(duration, hole.owner_end + CLIP_PADDING_SECONDS)
        segments = owned_segments(transcribe(clip_start, clip_end), hole, offset=clip_start)
        recovered.extend(segments)
        report.append(
            {
                **asdict(hole),
                "clipStart": clip_start,
                "clipEnd": clip_end,
                "recoveredText": "".join(str(segment["text"]) for segment in segments),
            }
        )
    provenance: dict[str, object] = {
        "frameSeconds": FRAME_SECONDS,
        "activeThresholdDBFS": active_threshold(levels),
        "minimumHoleSeconds": MINIMUM_HOLE_SECONDS,
        "holes": report,
        "recoveredSegments": len(recovered),
        "existingClocksUnchanged": True,
    }
    return merge(raw, recovered), provenance


def read_mono16(path: Path) -> array[int]:
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != SAMPLE_RATE
        ):
            raise ReplayError("Expected 16 kHz mono 16-bit recognition audio")
        samples = array("h")
        samples.frombytes(audio.readframes(audio.getnframes()))
    return samples


def clip_transcriber(
    samples: array[int], transcribe: Transcriber, directory: Path
) -> Callable[[float, float], object]:
    """Write each clip as its own WAV so the recognizer sees only that span."""

    def run(start: float, end: float) -> object:
        clip = directory / f"clip-{start:.1f}-{end:.1f}.wav"
        with wave.open(str(clip), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(SAMPLE_RATE)
            audio.writeframes(
                samples[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)].tobytes()
            )
        try:
            return transcribe(clip)
        finally:
            clip.unlink()

    return run


def main(argv: list[str] | None = None, *, transcribe: Transcriber | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="16 kHz mono 16-bit recognition WAV")
    parser.add_argument("raw", type=Path, help="full-pass recognition JSON")
    parser.add_argument("output", type=Path, help="completed recognition JSON")
    parser.add_argument("provenance", type=Path)
    args = parser.parse_args(argv)
    for path in (args.output, args.provenance):
        if path.exists() or path.is_symlink():
            raise ReplayError(f"Refusing to overwrite {path}")
    if transcribe is None:
        raise ReplayError("A pinned recognizer must be supplied by the recognition runtime")
    samples = read_mono16(args.audio)
    raw: object = json.loads(args.raw.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        completed, provenance = recover(
            raw, samples, clip_transcriber(samples, transcribe, Path(directory))
        )
    mlx_groups(completed, duration=len(samples) / SAMPLE_RATE)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(completed, stream, ensure_ascii=False, allow_nan=False, indent=2)
    with args.provenance.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
    holes = provenance["holes"]
    count = len(holes) if isinstance(holes, list) else 0
    print(json.dumps({"holes": count, "recovered": provenance["recoveredSegments"]}))
    return 0
