"""Correct caption speaker lanes only when two independent estimators agree.

A cue keeps its current speaker unless a diarizer reports exactly one speaker over
most of the cue *and* the loudest separated voice stem names that same speaker.
Ambiguous, overlapping or unassigned cues are never relabelled, and no speaker is
invented. Cue clocks are on the cut output timeline; diarization and stems are on
the uncut source timeline, mapped through the removed frame spans.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from array import array
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import STYLES, SpeakerCue, read_cues

FPS = 30
DOMINANT_COVERAGE = 0.8
FLOAT32_FORMAT = 3
MAX_SEGMENTS = 1_000_000


@dataclass(frozen=True)
class DiarizedSegment:
    start: float
    end: float
    speaker: int


class Float32Stem:
    """Read RMS over source-time windows from a mono 32-bit float WAV file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with path.open("rb") as stream:
            header = stream.read(12)
            if header[:4] != b"RIFF" or header[8:] != b"WAVE":
                raise ReplayError("Stem must be a RIFF/WAVE file")
            self.rate = 0
            self.offset = 0
            self.samples = 0
            while chunk := stream.read(8):
                name, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
                if name == b"fmt ":
                    body = stream.read(size)
                    form, channels, rate = struct.unpack("<HHI", body[:8])
                    bits = struct.unpack("<H", body[14:16])[0]
                    if form != FLOAT32_FORMAT or channels != 1 or bits != 32:
                        raise ReplayError("Stem must be mono 32-bit float PCM")
                    self.rate = rate
                elif name == b"data":
                    self.offset = stream.tell()
                    self.samples = size // 4
                    break
                else:
                    stream.seek(size + size % 2, 1)
        if not self.rate or not self.offset:
            raise ReplayError("Stem has no format or data chunk")

    def rms(self, start: float, end: float) -> float:
        first = max(0, math.floor(start * self.rate))
        last = min(self.samples, math.ceil(end * self.rate))
        if last <= first:
            return 0.0
        values = array("f")
        with self.path.open("rb") as stream:
            stream.seek(self.offset + first * 4)
            values.frombytes(stream.read((last - first) * 4))
        return math.sqrt(sum(value * value for value in values) / len(values))


def load_diarization(path: Path) -> tuple[DiarizedSegment, ...]:
    data = record(json.loads(path.read_text(encoding="utf-8")))
    segments = data.get("segments")
    if not isinstance(segments, list) or len(segments) > MAX_SEGMENTS:
        raise ReplayError("Diarization must contain a bounded segment list")
    result: list[DiarizedSegment] = []
    for item in segments:
        fields = record(item)
        start, end, speaker = fields.get("start"), fields.get("end"), fields.get("speaker")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or type(speaker) is not int
            or not 0 <= start < end
        ):
            raise ReplayError("Invalid diarization segment")
        result.append(DiarizedSegment(float(start), float(end), speaker))
    return tuple(result)


def source_seconds(output_seconds: float, cuts: tuple[tuple[int, int], ...]) -> float:
    """Map a cut-output time back to the source by re-adding removed frames."""
    frame = output_seconds * FPS
    for first, last in cuts:
        if first <= frame:
            frame += last - first + 1
    return frame / FPS


def dominant_speaker(start: float, end: float, segments: tuple[DiarizedSegment, ...]) -> int | None:
    """The only diarized speaker, if it covers most of the window."""
    coverage: dict[int, float] = {}
    for segment in segments:
        overlap = min(end, segment.end) - max(start, segment.start)
        if overlap > 0:
            coverage[segment.speaker] = coverage.get(segment.speaker, 0.0) + overlap
    if len(coverage) != 1:
        return None
    speaker, covered = next(iter(coverage.items()))
    return speaker if covered >= DOMINANT_COVERAGE * (end - start) else None


def agreed_lane(
    start: float,
    end: float,
    *,
    segments: tuple[DiarizedSegment, ...],
    diarized_lanes: dict[int, str],
    stems: dict[str, Float32Stem],
    cuts: tuple[tuple[int, int], ...],
) -> str | None:
    """The lane both estimators name for an output-time span, or None if they differ."""
    source_start, source_end = source_seconds(start, cuts), source_seconds(end, cuts)
    diarized = dominant_speaker(source_start, source_end, segments)
    lane = diarized_lanes.get(diarized) if diarized is not None else None
    if lane is None:
        return None
    loudest = max(stems, key=lambda name: stems[name].rms(source_start, source_end))
    return lane if loudest == lane else None


def reconcile(
    cues: tuple[SpeakerCue, ...],
    *,
    segments: tuple[DiarizedSegment, ...],
    diarized_lanes: dict[int, str],
    stems: dict[str, Float32Stem],
    cuts: tuple[tuple[int, int], ...],
) -> tuple[tuple[SpeakerCue, ...], Counter[tuple[str, str]]]:
    """Relabel a cue only when diarization and the loudest stem agree on another lane."""
    if not set(diarized_lanes.values()) <= set(STYLES) - {"unassigned"} or not stems:
        raise ReplayError("Lanes must be identified speaker lanes")
    changes: Counter[tuple[str, str]] = Counter()
    result: list[SpeakerCue] = []
    for cue in cues:
        if cue.speaker == "unassigned":
            result.append(cue)
            continue
        lane = agreed_lane(
            cue.start,
            cue.end,
            segments=segments,
            diarized_lanes=diarized_lanes,
            stems=stems,
            cuts=cuts,
        )
        if lane is None or lane == cue.speaker:
            result.append(cue)
            continue
        changes[(cue.speaker, lane)] += 1
        result.append(replace(cue, speaker=lane))
    return tuple(result), changes


def load_cut_frames(path: Path) -> tuple[tuple[int, int], ...]:
    """Removed inclusive frame spans of the cut output, as written beside the cues."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ReplayError("Cut frames must be a list")
    return tuple((int(item[0]), int(item[1])) for item in raw)


def key_values(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        key, separator, target = value.partition("=")
        if not separator or not key or not target:
            raise ReplayError("Expected KEY=VALUE")
        mapping[key] = target
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cues", type=Path)
    parser.add_argument("diarization", type=Path)
    parser.add_argument("cut_frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lane", action="append", default=[], help="DIARIZED_ID=lane")
    parser.add_argument("--stem", action="append", default=[], help="lane=voice.wav")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite reconciled cues")
    lanes = {int(key): lane for key, lane in key_values(args.lane).items()}
    stems = {lane: Float32Stem(Path(path)) for lane, path in key_values(args.stem).items()}
    cuts = load_cut_frames(args.cut_frames)
    cues, changes = reconcile(
        read_cues(args.cues),
        segments=load_diarization(args.diarization),
        diarized_lanes=lanes,
        stems=stems,
        cuts=cuts,
    )
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start", "end", "speaker", "text"])
        writer.writerows((cue.start, cue.end, cue.speaker, cue.text) for cue in cues)
    print(json.dumps({f"{old}->{new}": count for (old, new), count in sorted(changes.items())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
