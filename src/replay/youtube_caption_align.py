"""Give reviewed single-speaker cue text real word clocks, then caption like TikTok.

The reviewed cue text is kept. Words recognized inside a cue window are aligned to
that text; when the alignment is close, the text is split at recognized word
boundaries and each piece takes that word's clocks, so captions grow and switch
naturally. Cues that do not align keep their original single display: clocks are
never invented by dividing time across characters.
"""

from __future__ import annotations

import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

from replay.caption_io import mlx_groups
from replay.caption_timing import CaptionPolicy, SpokenToken, make_captions
from replay.core import ReplayError
from replay.speaker_subtitles import SpeakerCue, read_cues

SINGLE_POLICY = CaptionPolicy(maximum_characters=24, maximum_lines=2, tail_overrun=True)
MINIMUM_COVERAGE = 0.8
# A recognizer piece starting with these continues the previous word.
CONTINUATION = frozenset("ーぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ、。！？!?,.・…」』）)")
WINDOW_MARGIN = 0.3
# Captions shorter than a millisecond never reach a video frame or the CSV clock.
MINIMUM_DISPLAY = 0.001


def _compact(text: str) -> str:
    return "".join(text.split())


def _trim_to_match(text: str, words: list[SpokenToken]) -> list[SpokenToken]:
    """Drop neighbouring words outside the recognized span that matches the cue text."""
    recognized = "".join(_compact(word.text) for word in words)
    blocks = [
        block
        for block in SequenceMatcher(None, recognized, text, autojunk=False).get_matching_blocks()
        if block.size
    ]
    if not blocks:
        return []
    first, last = blocks[0].a, blocks[-1].a + blocks[-1].size
    kept: list[SpokenToken] = []
    position = 0
    for word in words:
        end = position + len(_compact(word.text))
        if end > first and position < last:
            kept.append(word)
        position = end
    return kept


def _join_continuations(pieces: list[SpokenToken]) -> list[SpokenToken]:
    joined: list[SpokenToken] = []
    for piece in pieces:
        if joined and piece.text[0] in CONTINUATION:
            previous = joined.pop()
            piece = SpokenToken(
                previous.text + piece.text, previous.start, piece.end, previous.utterance
            )
        joined.append(piece)
    return joined


def split_at_words(text: str, words: list[SpokenToken]) -> list[SpokenToken] | None:
    """Cut reviewed text at the recognized word boundaries, or None if they disagree."""
    target = _compact(text)
    words = _trim_to_match(target, words) if target else []
    recognized = "".join(_compact(word.text) for word in words)
    if not target or not recognized:
        return None
    matcher = SequenceMatcher(None, recognized, target, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    if matched < MINIMUM_COVERAGE * len(target):
        return None
    mapping = [0] * (len(recognized) + 1)
    for _tag, first, last, other_first, other_last in matcher.get_opcodes():
        for offset in range(first, last + 1):
            span = last - first
            mapping[offset] = other_first + (
                (offset - first) * (other_last - other_first) // span if span else 0
            )
    pieces: list[SpokenToken] = []
    cursor = 0
    position = 0
    for word in words:
        position += len(_compact(word.text))
        end = len(target) if word is words[-1] else mapping[position]
        if end > cursor:
            pieces.append(SpokenToken(target[cursor:end], word.start, word.end, word.utterance))
            cursor = end
        elif pieces:
            previous = pieces.pop()
            pieces.append(SpokenToken(previous.text, previous.start, word.end, previous.utterance))
    return _join_continuations(pieces)


def align(
    cues: tuple[SpeakerCue, ...], words: tuple[SpokenToken, ...]
) -> tuple[tuple[SpeakerCue, ...], int]:
    """Return TikTok-style cues and the number of reviewed cues that aligned."""
    result: list[SpeakerCue] = []
    aligned = 0
    for index, cue in enumerate(sorted(cues, key=lambda item: item.start)):
        inside = [
            SpokenToken(word.text, word.start, word.end, f"cue-{index}")
            for word in words
            if cue.start - WINDOW_MARGIN <= word.start < cue.end + WINDOW_MARGIN
        ]
        pieces = split_at_words(cue.text, inside)
        captions = None
        if pieces:
            try:
                captions = make_captions(tuple(pieces), policy=SINGLE_POLICY)
            except ValueError:
                captions = None
        if not captions:
            result.append(cue)
            continue
        aligned += 1
        result.extend(
            SpeakerCue(caption.start, caption.end, cue.speaker, caption.text)
            for caption in captions
            if caption.end - caption.start >= MINIMUM_DISPLAY
        )
    return tuple(result), aligned


def load_words(path: Path, duration: float) -> tuple[SpokenToken, ...]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    # Zero-length recognizer points carry no clock evidence and are not split points.
    return tuple(
        group.spoken_token()
        for group in mlx_groups(raw, duration=duration)
        if group.end > group.start
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cues", type=Path, help="reviewed start,end,speaker,text CSV")
    parser.add_argument("words", type=Path, help="MLX Whisper JSON with word timestamps")
    parser.add_argument("duration", type=float, help="recognized audio duration in seconds")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite aligned cues")
    cues = read_cues(args.cues)
    result, aligned = align(cues, load_words(args.words, args.duration))
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start", "end", "speaker", "text"])
        writer.writerows(
            (f"{cue.start:.3f}", f"{cue.end:.3f}", cue.speaker, cue.text) for cue in result
        )
    print(json.dumps({"reviewed": len(cues), "aligned": aligned, "cues": len(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
