"""Rebuild speaker captions from word tokens with the TikTok display policy.

Words accumulate until the line budget, a sentence end, a silence gap or a new
utterance, then a fresh caption starts, instead of a sliding window that drops one
word at a time. Every switch falls between recognized words, so no word is split.
Speakers come from reviewed cues matched by onset, which keeps prior relabels and
text corrections; cues without a matching token are kept as they are.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from replay.caption_timing import CaptionPolicy, SpokenToken, make_captions
from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import SpeakerCue, read_cues

LANE_POLICY = CaptionPolicy(maximum_characters=12, maximum_lines=2, tail_overrun=True)
ONSET_TOLERANCE = 0.01
# Captions shorter than a millisecond never reach a video frame or the CSV clock.
MINIMUM_DISPLAY = 0.001


def load_voice_tokens(path: Path, lanes: dict[int, str]) -> list[tuple[str, SpokenToken]]:
    """Read per-voice projected word tokens and their default lane."""
    data = record(json.loads(path.read_text(encoding="utf-8")))
    voices = data.get("voices")
    if not isinstance(voices, list):
        raise ReplayError("Candidate has no voice list")
    tokens: list[tuple[str, SpokenToken]] = []
    for item in voices:
        voice = record(item)
        number = voice.get("voice")
        projected = voice.get("projected_tokens")
        if type(number) is not int or number not in lanes or not isinstance(projected, list):
            raise ReplayError("Candidate voice is not mapped to a lane")
        for raw in projected:
            token = record(raw)
            text, start, end = token.get("text"), token.get("start"), token.get("end")
            utterance = token.get("utterance")
            if (
                not isinstance(text, str)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not isinstance(utterance, str)
            ):
                raise ReplayError("Invalid projected token")
            tokens.append(
                (lanes[number], SpokenToken(text.strip(), float(start), float(end), utterance))
            )
    return tokens


def _reviewed(lane: str, token: SpokenToken, cues: list[SpeakerCue]) -> tuple[str, SpokenToken]:
    """Take speaker and corrected text from the reviewed cue that ends with this word."""
    candidates = [cue for cue in cues if cue.speaker != "unassigned"]
    same_lane = next((cue for cue in candidates if cue.speaker == lane), None)
    only = candidates[0] if len(candidates) == 1 else None
    match = next(
        (cue for cue in candidates if cue.text.replace("\n", "").endswith(token.text)),
        same_lane or only,
    )
    if match is None:
        return lane, token
    reviewed = match.text.replace("\n", "")
    # A reviewed correction replaces the recognized word with text of the same length.
    text = token.text if reviewed.endswith(token.text) else reviewed[-len(token.text) :]
    return match.speaker, SpokenToken(text, token.start, token.end, token.utterance)


def _merge_shared_onsets(words: list[SpokenToken]) -> tuple[SpokenToken, ...]:
    """Words of one lane that start together are shown as one word, never reordered."""
    merged: list[SpokenToken] = []
    for word in sorted(words, key=lambda item: (item.start, item.end)):
        if merged and merged[-1].start == word.start:
            previous = merged.pop()
            word = SpokenToken(
                previous.text + word.text,
                previous.start,
                max(previous.end, word.end),
                previous.utterance,
            )
        merged.append(word)
    return tuple(merged)


def rebuild(
    tokens: list[tuple[str, SpokenToken]], reviewed: tuple[SpeakerCue, ...]
) -> tuple[SpeakerCue, ...]:
    """Group word tokens per reviewed speaker and caption them like TikTok."""
    lanes: dict[str, list[SpokenToken]] = {}
    for lane, token in tokens:
        window = [cue for cue in reviewed if abs(cue.start - token.start) <= ONSET_TOLERANCE]
        speaker, corrected = _reviewed(lane, token, window)
        lanes.setdefault(speaker, []).append(
            SpokenToken(
                corrected.text, corrected.start, corrected.end, f"{speaker}:{corrected.utterance}"
            )
        )
    cues: list[SpeakerCue] = []
    for speaker, words in lanes.items():
        for caption in make_captions(_merge_shared_onsets(words), policy=LANE_POLICY):
            if caption.end - caption.start >= MINIMUM_DISPLAY:
                cues.append(SpeakerCue(caption.start, caption.end, speaker, caption.text))
    cues.extend(cue for cue in reviewed if cue.speaker == "unassigned")
    return tuple(sorted(cues, key=lambda cue: (cue.start, cue.speaker)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="JSON with voices[].projected_tokens")
    parser.add_argument("reviewed", type=Path, help="reviewed start,end,speaker,text CSV")
    parser.add_argument("output", type=Path)
    parser.add_argument("--lane", action="append", default=[], help="VOICE=lane")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite rebuilt cues")
    lanes: dict[int, str] = {}
    for value in args.lane:
        voice, separator, lane = value.partition("=")
        if not separator or not voice.isdigit() or not lane:
            raise ReplayError("Expected VOICE=lane")
        lanes[int(voice)] = lane
    cues = rebuild(load_voice_tokens(args.candidate, lanes), read_cues(args.reviewed))
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start", "end", "speaker", "text"])
        writer.writerows(
            (f"{cue.start:.3f}", f"{cue.end:.3f}", cue.speaker, cue.text) for cue in cues
        )
    print(json.dumps({"cues": len(cues)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
