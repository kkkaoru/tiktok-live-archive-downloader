"""Caption YouTube speech that is clearly audible but has no caption cue.

The separated vocals, already on the cut output timeline, are measured against the
existing cues. Every voiced span of at least two seconds without a cue is
re-recognized as a clip with the pinned local model (see replay.asr_gap_recovery);
only words lying wholly inside the span are kept. Recovered words are captioned
with the same growing TikTok-style policy as the other speaker cues. A recovered
caption gets a speaker lane only when diarization and the loudest separated stem
agree on it; otherwise it stays unassigned. Existing cues are never changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from array import array
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from replay.asr_gap_recovery import (
    CLIP_PADDING_SECONDS,
    SAMPLE_RATE,
    Transcriber,
    clip_transcriber,
    frame_levels,
    owned_segments,
    read_mono16,
    voiced_holes,
)
from replay.caption_timing import SpokenToken, make_captions
from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_reconcile import (
    Float32Stem,
    agreed_lane,
    key_values,
    load_cut_frames,
    load_diarization,
)
from replay.speaker_subtitles import SpeakerCue, read_cues
from replay.youtube_caption_rebuild import LANE_POLICY, MINIMUM_DISPLAY

UNASSIGNED = "unassigned"

Labeller = Callable[[float, float], str]


def segment_tokens(segment: object, utterance: str) -> tuple[SpokenToken, ...]:
    """Recovered words with real clocks; zero-length points carry no clock evidence."""
    words = record(segment).get("words")
    tokens: list[SpokenToken] = []
    for item in words if isinstance(words, list) else []:
        word = record(item)
        text, start, end = word.get("word"), word.get("start"), word.get("end")
        if (
            isinstance(text, str)
            and text.strip()
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and end > start
        ):
            tokens.append(SpokenToken(text.strip(), float(start), float(end), utterance))
    return tuple(tokens)


def recover(
    cues: tuple[SpeakerCue, ...],
    samples: array[int],
    transcribe: Callable[[float, float], object],
    label: Labeller,
) -> tuple[tuple[SpeakerCue, ...], dict[str, object]]:
    """Return the cues with recovered captions added, and the provenance of each hole."""
    holes = voiced_holes(frame_levels(samples), sorted((cue.start, cue.end) for cue in cues))
    duration = len(samples) / SAMPLE_RATE
    added: list[SpeakerCue] = []
    report: list[dict[str, object]] = []
    for index, hole in enumerate(holes):
        clip_start = max(0.0, hole.owner_start - CLIP_PADDING_SECONDS)
        clip_end = min(duration, hole.owner_end + CLIP_PADDING_SECONDS)
        segments = owned_segments(transcribe(clip_start, clip_end), hole, offset=clip_start)
        texts: list[str] = []
        for number, segment in enumerate(segments):
            tokens = segment_tokens(segment, f"recovered:{index}:{number}")
            if not tokens:
                continue
            speaker = label(tokens[0].start, tokens[-1].end)
            for caption in make_captions(tokens, policy=LANE_POLICY):
                if caption.end - caption.start >= MINIMUM_DISPLAY:
                    added.append(SpeakerCue(caption.start, caption.end, speaker, caption.text))
            texts.append("".join(token.text for token in tokens))
        report.append({**asdict(hole), "recoveredText": "".join(texts)})
    provenance: dict[str, object] = {"holes": report, "recoveredCues": len(added)}
    combined = sorted((*cues, *added), key=lambda cue: (cue.start, cue.speaker))
    return tuple(combined), provenance


def lane_labeller(
    *,
    diarization: Path,
    cut_frames: Path,
    lanes: dict[int, str],
    stems: dict[str, Float32Stem],
) -> Labeller:
    """Name a lane only when diarization and the loudest stem agree."""
    segments = load_diarization(diarization)
    cuts = load_cut_frames(cut_frames)

    def label(start: float, end: float) -> str:
        lane = agreed_lane(
            start, end, segments=segments, diarized_lanes=lanes, stems=stems, cuts=cuts
        )
        return lane if lane is not None else UNASSIGNED

    return label


def main(argv: list[str] | None = None, *, transcribe: Transcriber | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cues", type=Path, help="start,end,speaker,text CSV on the output timeline")
    parser.add_argument(
        "vocals", type=Path, help="16 kHz mono 16-bit vocals on the output timeline"
    )
    parser.add_argument("diarization", type=Path)
    parser.add_argument("cut_frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("--lane", action="append", default=[], help="DIARIZED_ID=lane")
    parser.add_argument("--stem", action="append", default=[], help="lane=voice.wav")
    args = parser.parse_args(argv)
    for path in (args.output, args.provenance):
        if path.exists() or path.is_symlink():
            raise ReplayError(f"Refusing to overwrite {path}")
    if transcribe is None:
        raise ReplayError("A pinned recognizer must be supplied by the recognition runtime")
    label = lane_labeller(
        diarization=args.diarization,
        cut_frames=args.cut_frames,
        lanes={int(key): lane for key, lane in key_values(args.lane).items()},
        stems={lane: Float32Stem(Path(path)) for lane, path in key_values(args.stem).items()},
    )
    samples = read_mono16(args.vocals)
    with tempfile.TemporaryDirectory() as directory:
        cues, provenance = recover(
            read_cues(args.cues),
            samples,
            clip_transcriber(samples, transcribe, Path(directory)),
            label,
        )
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start", "end", "speaker", "text"])
        writer.writerows(
            (f"{cue.start:.3f}", f"{cue.end:.3f}", cue.speaker, cue.text) for cue in cues
        )
    with args.provenance.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"cues": len(cues), "recovered": provenance["recoveredCues"]}))
    return 0
