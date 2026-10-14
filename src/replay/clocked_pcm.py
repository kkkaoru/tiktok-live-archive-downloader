"""Write PCM supports with explicit clocks, never serialize timestamp holes away.

This is a sparse intermediate, not a playable full delivery. Missing support has
no waveform and is not silence. Consumers must verify the supplied native clock
correspondence before applying any playback, mixing or resampling policy.
"""

import json
import os
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.caption_measured_job import ROOT, _local
from replay.continuous_cuts import write_pcm_spans
from replay.delivery_audio_mapping import integer
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import _digest

MAX_SPANS = 100_000
MAX_SECONDS = 6 * 3600


@dataclass(frozen=True, slots=True)
class ClockedSpan:
    source_start_sample: int
    sample_count: int
    source_time: Fraction
    output_time: Fraction


def validate_placements(
    spans: tuple[ClockedSpan, ...], *, sample_rate: int, source_frames: int, end: Fraction
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Reject ambiguous overlaps and fractional sample placement; retain holes."""
    if (
        type(sample_rate) is not int
        or not 8000 <= sample_rate <= 192000
        or type(source_frames) is not int
        or not 0 < source_frames <= MAX_SECONDS * sample_rate
        or not isinstance(end, Fraction)
        or not 0 < end <= MAX_SECONDS
        or (end * sample_rate).denominator != 1
        or not 1 <= len(spans) <= MAX_SPANS
    ):
        raise ValueError("Invalid clocked PCM budget or terminal sample clock")
    holes: list[tuple[Fraction, Fraction]] = []
    previous_end = Fraction(0)
    previous_source_end = 0
    previous_native_end = Fraction(0)
    for span in spans:
        if (
            type(span.source_start_sample) is not int
            or type(span.sample_count) is not int
            or span.source_start_sample < previous_source_end
            or span.sample_count <= 0
            or span.source_start_sample + span.sample_count > source_frames
            or not isinstance(span.source_time, Fraction)
            or not isinstance(span.output_time, Fraction)
            or span.source_time < previous_native_end
            or span.output_time < previous_end
            or (span.source_time * sample_rate).denominator != 1
            or (span.output_time * sample_rate).denominator != 1
        ):
            raise ValueError("Ambiguous overlap, fractional clock or invalid source support")
        duration = Fraction(span.sample_count, sample_rate)
        if span.output_time + duration > end:
            raise ValueError("PCM support exceeds declared output extent")
        if span.output_time > previous_end:
            holes.append((previous_end, span.output_time))
        previous_end = span.output_time + duration
        previous_source_end = span.source_start_sample + span.sample_count
        previous_native_end = span.source_time + duration
    if previous_end < end:
        holes.append((previous_end, end))
    return tuple(holes)


def reader_placements(
    observation: object,
    *,
    sample_rate: int,
    source_frames: int,
    origin: Fraction,
    input_clock: str,
) -> tuple[ClockedSpan, ...]:
    """Bind reader PCM ordinals to returned presentation buffers, not packet PTS.

    The input must be an unchanged, independently verified PCM readback of this
    reader execution. This does not bind original continuous PCM or model output.
    Calling this again on already projected data is forbidden. Request-dependent
    readers may give the same content different clocks; do not combine runs.
    """
    if input_clock != "reader-pcm-ordinal" or not isinstance(origin, Fraction):
        raise ValueError("Reader mapping requires unprojected reader PCM ordinals")
    data = record(observation)
    buffers = data.get("buffers")
    if data.get("mode") not in ("track", "audio-mix") or not isinstance(buffers, list):
        raise ValueError("Expected actual presentation reader buffers, not packet metadata")
    if not 1 <= len(buffers) <= 10000 or not 8000 <= sample_rate <= 192000:
        raise ValueError("Invalid reader buffer budget or sample rate")
    spans: list[ClockedSpan] = []
    ordinal = 0
    for item in buffers:
        row = record(item)
        pts = Fraction(integer(row.get("value")), integer(row.get("timescale"), minimum=1))
        duration = Fraction(
            integer(row.get("durationValue"), minimum=1),
            integer(row.get("durationTimescale"), minimum=1),
        )
        count = integer(row.get("samples"), minimum=1)
        if duration != Fraction(count, sample_rate):
            raise ValueError("Reader duration and actual PCM sample support disagree")
        spans.append(ClockedSpan(ordinal, count, pts, pts - origin))
        ordinal += count
    if ordinal != source_frames:
        raise ValueError("PCM readback sample count differs from observed reader")
    end = spans[-1].output_time + Fraction(spans[-1].sample_count, sample_rate)
    validate_placements(tuple(spans), sample_rate=sample_rate, source_frames=source_frames, end=end)
    return tuple(spans)


def observed_pcm_mov_command(
    source: Path,
    output: Path,
    *,
    spans: tuple[ClockedSpan, ...],
    ffmpeg: Path,
) -> tuple[str, ...]:
    """Make only a complete contiguous reader support playable, without recoding.

    The MOV movie clock uses the observed PCM sample grid so a default 600Hz edit
    cannot shorten its tail. Sparse supports, skipped samples and output gaps are
    refused, not concatenated, filled or trimmed. This does not repair an entire
    requested interval whose reader returned less support.
    """
    source, output = _local(str(source.absolute())), _local(str(output.absolute()))
    if not output.is_relative_to(ROOT / "edits/work") or output.exists():
        raise ValueError("Playable diagnostic requires a new repository-local output")
    with wave.open(str(source), "rb") as reader:
        rate, frames = reader.getframerate(), reader.getnframes()
        if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise ValueError("Playable diagnostic requires PCM16 readback")
    end = Fraction(frames, rate)
    holes = validate_placements(spans, sample_rate=rate, source_frames=frames, end=end)
    if (
        holes
        or any(
            span.output_time != Fraction(span.source_start_sample, rate)
            or span.source_time != spans[0].source_time + Fraction(span.source_start_sample, rate)
            for span in spans
        )
        or sum(span.sample_count for span in spans) != frames
    ):
        raise ValueError("Playable diagnostic cannot concatenate sparse or skipped support")
    return (
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-n",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-c:a",
        "copy",
        "-movie_timescale",
        str(rate),
        "-f",
        "mov",
        str(output),
    )


def write_clocked_pcm(
    source: Path,
    directory: Path,
    *,
    source_sha256: str,
    spans: tuple[ClockedSpan, ...],
    end: Fraction,
) -> Path:
    """Copy each support separately and bind exact source/output rational clocks.

    No hole bytes, mix, trim, resampler, model or inferred native clock is created.
    The caller owns independent correspondence evidence. New local output only;
    failed partials remain and source/code hashes are rechecked before publication.
    """
    source = _local(str(source.absolute()))
    directory = _local(str(directory.absolute()))
    if not directory.is_relative_to(ROOT / "edits/work"):
        raise ValueError("Clocked PCM output must remain in checkout edits/work")
    digest = _digest(source_sha256)
    if file_digest(source) != digest:
        raise ValueError("Source PCM digest differs")
    code = {str(Path(__file__).resolve()): file_digest(Path(__file__))}
    copy_code = ROOT / "src/replay/continuous_cuts.py"
    code[str(copy_code)] = file_digest(copy_code)
    with wave.open(str(source), "rb") as reader:
        rate, frames = reader.getframerate(), reader.getnframes()
        if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise ValueError("Clocked PCM requires PCM16")
        holes = validate_placements(spans, sample_rate=rate, source_frames=frames, end=end)
    directory.mkdir(mode=0o700)
    pieces: list[dict[str, object]] = []
    for index, span in enumerate(spans):
        target = directory / f"support-{index:06d}.wav"
        write_pcm_spans(
            source,
            target,
            ((span.source_start_sample, span.source_start_sample + span.sample_count),),
        )
        target.chmod(0o600)
        pieces.append(
            {
                "path": str(target),
                "sha256": file_digest(target),
                "sourceStartSample": span.source_start_sample,
                "sampleCount": span.sample_count,
                "sourceTime": str(span.source_time),
                "outputTime": str(span.output_time),
                "outputStartSample": int(span.output_time * rate),
                "duration": str(Fraction(span.sample_count, rate)),
            }
        )
    if file_digest(source) != digest or any(
        file_digest(Path(path)) != pin for path, pin in code.items()
    ):
        raise ValueError("Source PCM or writer implementation changed")
    receipt = directory / "placement.json"
    with receipt.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(
            {
                "version": 1,
                "kind": "sparse-clocked-pcm",
                "source": {"path": str(source), "sha256": digest},
                "implementation": code,
                "sampleRate": rate,
                "outputEnd": str(end),
                "pieces": pieces,
                "missingSupport": [[str(start), str(stop)] for start, stop in holes],
                "missingSupportPolicy": "unmaterialized-not-observed-silence",
                "playbackPolicy": None,
                "sourceCorrespondenceVerified": False,
                "deliveryApproved": False,
            },
            handle,
            indent=2,
        )
    return receipt
